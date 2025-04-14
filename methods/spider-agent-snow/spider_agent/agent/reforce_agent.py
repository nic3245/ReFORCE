from spider_agent.agent.agents import PromptAgent
from spider_agent.envs.table_compression import compress_tables, compress_ddl_files

from typing import Dict, List, Any, Optional, Tuple
import threading
import pandas as pd
import logging
import concurrent.futures
import json
import re
from io import StringIO
import numpy as np

from spider_agent.agent.prompts import SNOWFLAKE_REFORCE_SYSTEM, EXPLORATION_REFORCE_SYSTEM
from spider_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS
from spider_agent.agent.models import call_llm

logger = logging.getLogger("spider_agent")

class ReFoRCEAgent(PromptAgent):
    """
    Implementation of the ReFoRCE (Self-Refinement Agent with Format 
    Restriction and Column Exploration) approach.
    
    This agent enhances the standard Spider Agent with:
    1. Table compression to mitigate long-context limitations
    2. Format restriction to ensure accurate answer format
    3. Iterative column exploration for enhanced schema understanding
    4. Self-refinement pipeline with parallelized workflows and voting
    5. CTE-based refinement for handling unresolved cases
    """
    
    def __init__(
        self,
        model="qwq-32b",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        parallel_count=3,  # Number of parallel threads
        max_refinement_iterations=5,  # Max iterations for self-refinement
        use_plan=False
    ):
        super().__init__(model, max_tokens, top_p, temperature, 
                          max_memory_length, max_steps, use_plan)
        
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Auto-detect display width
        pd.set_option('display.max_colwidth', None)  # Show full content in each cell
        self.parallel_count = parallel_count
        self.max_refinement_iterations = max_refinement_iterations
        self.results_cache = {}  # Cache for storing execution results
        self.format_specs = {}   # Store format specifications
        
    def set_env_and_task(self, env):
        """
        Override the parent method to include table compression and format restriction setup.
        """
        super().set_env_and_task(env)
        # REDO SYSTEM MESSAGE STUFF B/C WE WANT TO USE THE ONE WE MADE
        self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, SNOWFLAKE_EXEC_SQL, CreateFile, EditFile]
        action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        self.system_message = SNOWFLAKE_REFORCE_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })

    def _get_llm_response(self, prompt: str, header: str, history_messages: List[Dict]) -> str:
        status = False
        while not status:
            messages = history_messages.copy()
            messages.append({
                "role": "user", # XXX idk if this works
                "content": [
                    {
                        "type": "text",
                        "text": f"{str(header)}: {str(prompt)}\n"
                    }
                ]
            })  
            status, response = call_llm({
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "temperature": self.temperature
            })
            response = response.strip()
            if not status:
                if response in ["context_length_exceeded","rate_limit_exceeded","max_tokens","unknown_error"]:
                    history_messages = [history_messages[0]] + history_messages[3:]
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")
        return response

    
    def _read_file_content(self, file_path: str):
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content of the file or None if error
        """

        # Either simply cat it, or it's a json or csv that's readable
        obs, _ = self.env.step(Bash(code=f"cat {file_path}"))
        
        try:
            if file_path.endswith('.json'):
                return json.loads(obs)
            elif file_path.endswith('.csv'):
                return pd.read_csv(StringIO(obs))
            return obs
        except Exception as e:
            iter = 0
            copy = obs[:]
            
            logger.error(f"Error reading file {file_path}: {e}")
            if isinstance(e, json.JSONDecodeError):
                flag = True
                position = e.pos
                while flag and iter < 10000:
                    try: 
                        obs = obs[:position] + ' ' + obs[position+1:]
                        obs = json.loads(obs)
                        flag = False
                    except json.JSONDecodeError as e:
                        position = e.pos
                        iter += 1
                
            if iter >= 10000:
                return copy
            else:
                return obs
       
    def _get_raw_schema(self) -> Dict:
        """
        Extract schema information from the environment.
        
        Returns:
            Dict: Schema information
        """
        # This implementation would depend on how schema is stored in the environment
        # For Spider-Agent, we'll need to parse from files
        schema_info = {}
        
        # Execute bash commands to gather schema information
        obs, _ = self.env.step(Bash(code="find . -name '*.json' -o -name '*.csv' | grep -v 'result.json' | grep -v 'results.csv'"))
        
        # Parse the output and extract schema information
        for line in obs.split('\n'):
            if '.json' in line or '.csv' in line:
                file_path = line.strip()
                content = self._read_file_content(file_path)
                if content:
                    schema_info[file_path] = content

        logger.info(f"Schema extracted files: {list(schema_info.keys())[:min(5, len(schema_info.keys()))]}")
        
        return schema_info
    
    def _compress_schema(self, schema_info: Dict) -> Dict:
        """
        Apply table compression to the database schema to reduce context size.
        
        Returns:
            Dict: Compressed schema information
        """
        # Apply compression function
        compressed_schema = compress_tables(schema_info)

        logger.info(f"Compressed schema: {compressed_schema}")
        
        logger.info(f"Schema compressed from {len(str(schema_info))} to {len(str(compressed_schema))} characters")
        
        return compressed_schema
    
    def _generate_format_restrictions(self, compressed_schema, max_tries=5) -> Dict:
        """
        Generate expected answer format based on task instruction and schema. 
        This is done by giving a separate chat session and format prompt.
        It then gives you a csv or smth i don't know that yet with the necessary columns and types and stuff.
        That will be used/enforced later in self-refinement.
        
        Returns:
            Dict: Format specification
        """
        self.format_history = []
        format_prompt =  f"""################### INSTRUCTIONS ###################
Your goal is to generate a format specification for the expected answer based on the task instruction and the database schema. This should be a CSV where each column should be explicitly defined, including all necessary attributes.
The format should account for specific cases, such as superlatives, percentages, or coordinates, ensuring the output is concise, clear, and unambiguous. For ambiguous terms, potential values or additional columns can be added to maintain clarity and precision. 
Your response should be prefixed by "Format Specification: ". The CSV should have two lines, with the first line containing column names. The last column should always be num_rows, and its value in the second should be the number of rows in the expected output. -1 should be used to indicate any number of rows.
The values for the other columns in the second row should be their data type.
################### EXAMPLE ###################
Example task: Identify the case barcodes from the TCGA - LAML study with the highest weighted average copy number in cytoband 15q11 on chromosome 15, using segment data and cytoband overlaps from TCGA's genomic and Mitelman databases.
Format Specification: 
case_barcode,weighted_average_copy_number, num_rows
str, float, 1
################### SCHEMA ###################
This is a compressed database schema with representative tables:
{compressed_schema}
################### TASK ###################
Task:
{self.instruction}
"""
        flag = True
        tries = 0
        while flag and tries <= max_tries:
            response = self._get_llm_response(format_prompt, "", self.format_history)
            logger.info(f"LLM response for format specification: {response}")
            tries += 1
            try: 
                match = re.finditer(r"Format Specification:\s*(.*?)(?:'|\"|$|\n\n)", response, re.DOTALL)
                if match:
                    result = list(match)[-1].group(1)
                    format_spec = pd.read_csv(StringIO(result))
                else:
                    raise ValueError("No format specification prefix.")
                if 'num_rows' in format_spec.columns:
                    if 'num_rows' in format_spec.select_dtypes(include=[np.number]).columns:
                        flag = False
                    else:
                        flag = True
                        # TODO fix history appending
                        self.format_history.extend([{"role": "assistant", "content": [{"type": "text", "text": response}]}, {"role": "user", "content": [{"type": "text", "text": "You produced a valid CSV with a num_rows column, but its value was not a number. Try again, and make sure your CSV's num_rows column is either -1 (any number of rows) or the actual number of rows that the answer would contain."}]}])
                else: # no num_rows column, try again
                    flag = True
                    # TODO fix history appending
                    self.format_history.extend([{"role": "assistant", "content": [{"type": "text", "text": response}]}, {"role": "user", "content": [{"type": "text", "text": "You produced a valid CSV, but it did not contain a num_rows column. Try again, and make sure your CSV has a num_rows column."}]}])
            except Exception as e: # invalid csv, try again
                logger.error(f"Error parsing format specification: {e}")
                flag = True
                # TODO fix history appending
                self.format_history.extend([{"role": "assistant", "content": [{"type": "text", "text": response}]}, {"role": "user", "content": [{"type": "text", "text": "Failed to parse CSV from your response. Make sure your answer is prefixed with \"Format Specification: \" and is a valid CSV."}]}])
        logger.info(f"Generated format specification: {format_spec}")
        
        return format_spec
    
    def _generate_initial_prompt(self, compressed_schema, format_spec) -> str:
        """
        Generate initial prompt for the agent.
        
        Returns:
            str: Initial prompt
        """

        # Use the prompt generation module to generate prompt
        initial_prompt = f"""# COMPRESSED SCHEMA #
        {compressed_schema}
        # EXPECTED FORMAT #
        # {format_spec}"""
        
        logger.info(f"Generated initial prompt: {initial_prompt}")
        
        return initial_prompt
    
    def _parse_llm_response(self, response, pattern):
        matches = re.findall(pattern, response)
        return matches

    
    def _explore_columns(self, prompt: str) -> Dict:
        """
        Apply column exploration to the database schema.
        Column exploration is done by giving a separate chat session an exploration prompt and getting a list of sql queries.
        It then executes the queries and returns the results to the agent, which can either keep exploring or quit.
        Returns:
            Dict: Column information
        """
        self.correction_messages = []
        exploration_prompt = f"""
        {prompt}

        ######## COLUMN EXPLORATION ########
        Your task is to prepare for answering the user's question by exploring the columns in the database schema in depth. Use the provided compressed schema and all relevant information. 
        Please generate 1 new SQL query each time.

        ### Instructions:
        Generate a SQL query that will help you:
        1. Identify **relevant tables and columns** for the task.
        2. Retrieve **data types** and **example values** from these columns.
        3. Understand **nested structures**, if any, by using dialect-specific tools like `LATERAL FLATTEN` for JSON or nested types.
        4. Observe **column value distributions**, including distinct values, value ranges, or representative samples.

        ### Guidelines:
        - Use the Snowflake SQL dialect.
        - Start with **SELECT queries** to inspect a few rows from each relevant table.
        - Progress to **more focused queries** using `SELECT DISTINCT`, basic filtering (`LIKE`, `LIMIT`, etc.).
        - Include **exploration of nested or JSON fields** using dialect-specific constructs.
        - Each query should be concise and interpretable on its own, different from the previous queries.
        - Avoid using complex CTEs or joins at this stage. Focus on **schema and data understanding**.

        Provide the single query output in this format. 
        ```sql
        <your query here>
        ```
        """
        
        exploration_history = []
        exploration_history.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": EXPLORATION_REFORCE_SYSTEM
                },
            ]
        })
        sql_queries = []
        i = 0
        sql_pattern = r"```sql\r?\n([\s\S]*?)\r?\n```"
        while len(sql_queries) < 5:
            output = self._get_llm_response(exploration_prompt, "Column Exploration", exploration_history)
            logger.info(f"LLM Output {i}: {output}")

            history_string = ""
            matches = self._parse_llm_response(output, sql_pattern)
            if matches:
                logger.info(f"Found Match!{len(sql_queries)}")
                sql_queries.append(matches[0])
                # set history string to appropriate msg 
                history_string = f"""
                An exploration query was found successfully in the output you provided: 
                '''sql
                {matches[0]}
                '''
                When prompted to give the next query, please make the query different from ALL previously generated queries. 
                """
            else:
                # set history string to appropriate msg 
                history_string = f"""
                An exploration query was not found in the output you provided: 
                {output}
                
                ### FIX: 
                Please follow ALL instructions and guidelines clearly given regarding column exploration.
                Verify the final output is given in this format. 
                ```sql
                    <your query here>
                ```
                """
            exploration_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Exploration {i}: {history_string}\n"
                    }
                ]
            })  
            i += 1

        def algorithm_one(sql_actions):
            def self_correct(sql_repr, result):
                correction_prompt = f"""
                    You are an expert SQL assistant. You will be given:
                    - An incorrect user generated SQL query
                    - The results from running that query

                    User Generated SQL Query:
                    {sql_repr}

                    Result:
                    {result}

                    Your task is to **correct the SQL query** so that it runs correctly and returns valid, structured results. 
                    Use all previous information provided to you and your expert SQL knowledge to correct the query. 

                    Here are your goals:
                    1. Fix all syntax errors in the SQL query.
                    2. Logically verify that the query can return the intended correct results.
                    3. Ensure the query returns a non-empty, meaningful result with all columns containing valid data.
                    4. Avoid columns with all NULL values or obviously invalid data types.
                    5. Use context of previous queries to guide the correction.
                    
                    Respond with **only one corrected SQL query**. Do not include any explanation or comments.
                    Provide the final output of the corrected query in this format. 
                    ```sql
                        <your query here>
                    ```
                """

                header = "SQL Correction Task"
                correction_response = self._get_llm_response(correction_prompt, header, self.correction_messages)
                corrected_sql = self._parse_llm_response(correction_response, sql_pattern)

                correction_result = f"""
                    User Generated SQL Query:
                    {sql_repr}

                    The following correction was applied to the User Generated SQL Query:

                    Corrected SQL Query:
                    {correction_response}

                    Please use this as context for future related queries.
                """

                message = correction_prompt + correction_result
                self.correction_messages.append({
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"{message}"
                        }
                    ]
                }) 
                if corrected_sql:
                    return corrected_sql[-1]
                return "None"
            

            result_dic = {}
            error_rec = 0

            while len(sql_actions) > 0:
                sql_action = sql_actions.pop(0)
                logger.info(type(sql_action))
                result, _ = self.env.step(sql_action)
                try:
                    logger.info(f"Result: {result}")
                    match = re.search(r"(?:.*?warn_incompatible_dep\(\s*)?(\s*[A-Z_]+(?:\s+[A-Z_]+)*\s*\n(?:\s*\d+.*\n?)+)", result, re.DOTALL)
                    if not match:
                        result_dic[sql_action.__repr__()] = re.search(r"(?:.*?warn_incompatible_dep\(\s*\n?\s*)?(.+(?:\n.+)*)", result, re.DOTALL).group(1).strip()
                        raise ValueError("Could not find table text in the given string.")

                    table_str = match.group(1).strip()
                    df = pd.read_csv(StringIO(table_str), sep=r"\s+")
                    empty_cols = df.columns[df.isnull().all()]
                    snowflake_pattern = re.compile(r'sql_query\s*=\s*"([^"]+)"')
                    snowflake_match = snowflake_pattern.search(sql_action.__repr__())
                    if len(empty_cols) != 0:
                        message = f"""
                            The following SQL query contained empty columns: {list(empty_cols)}

                            Query:
                            {snowflake_match.group(1)}

                            Result (first 5 or fewer rows):
                            {df.head(5).to_string(index=False)}

                            This query is incorrected needs to be revised. Please remember this query and try to avoid the empty columns for future generated queries.
                            """
                        self.correction_messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"{message}"
                                }
                            ]
                        })
                        result_dic[sql_action.__repr__()] = df.to_dict() 
                        raise ValueError(f"Encountered empty columns: {list(empty_cols)}")
                    logger.info(f"{result_dic}")
                    result_dic[sql_action.__repr__()] = df.to_dict()
                    error_rec = 0
                    message = f"""
                        The following SQL query executed successfully and returned correct results:

                        Query:
                        {sql_action.__repr__()}

                        Result (first 5 or fewer rows):
                        {df.head(5).to_string(index=False)}

                        This query is excellent and does not need need to be revised. Please remember this query and its result as context to assist with future corrections.
                        """
                    self.correction_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{message}"
                            }
                        ]
                    })  
                    continue
                except: 
                # correction
                    max_iter = 3
                    # simplify = False
                    corrected_sql = None

                    for i in range(max_iter):
                        sql_repr = sql_action.__repr__()
                        corrected_sql = self_correct(sql_repr, result)
                        if corrected_sql == "None":
                            continue

                        new_action = SNOWFLAKE_EXEC_SQL(corrected_sql, is_save=False)
                        result, _ = self.env.step(new_action)
                        logger.info(result)
                        try:
                            match = re.search(r"(?:.*?warn_incompatible_dep\(\s*)?(\s*[A-Z_]+(?:\s+[A-Z_]+)*\s*\n(?:\s*\d+.*\n?)+)", result, re.DOTALL)
                            if not match:
                                result_dic[sql_action.__repr__()] = re.search(r"(?:.*?warn_incompatible_dep\(\s*\n?\s*)?(.+(?:\n.+)*)", result, re.DOTALL).group(1).strip()
                                raise ValueError("Could not find table text in the given string.")

                            table_str = match.group(1)
                            corrected_df = pd.read_csv(StringIO(table_str), sep=r"\s+")
                            corrected_empty_cols = corrected_df.columns[df.isnull().all()]
                            corrected_snowflake_pattern = re.compile(r'sql_query\s*=\s*"([^"]+)"')
                            corrected_snowflake_match = corrected_snowflake_pattern.search(sql_action.__repr__())
                            if len(empty_cols) != 0:
                                message = f"""
                                The following corrected SQL query contained empty columns: {list(corrected_empty_cols)}

                                Query:
                                {corrected_snowflake_match.group(1)}

                                Result (first 5 or fewer rows):
                                {corrected_df.head(5).to_string(index=False)}

                                This query is incorrected needs to be revised. Please remember this query and try to avoid the empty columns for future generated queries.
                                """
                                self.correction_messages.append({
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"{message}"
                                        }
                                    ]
                                }) 
                                result_dic[sql_action.__repr__()] = corrected_df.to_dict()
                                raise ValueError(f"Encountered empty columns: {list(empty_cols)}")
                        except:
                            continue
                        logger.info(f"{result_dic}")
                        result_dic[sql_action.__repr__()] = corrected_df.to_dict()
                        error_rec = 0

                        # apply correction to rest of sql_actions
                        for i in range(len(sql_actions)):
                            next_sql_repr = sql_actions[i].__repr__()
                            next_corrected_sql = self_correct(next_sql_repr, result)
                            if not next_corrected_sql == "None":
                                sql_actions[i] = next_corrected_sql 
                        break
                    error_rec += 1
                if error_rec > 5:
                    return result_dic
            return result_dic
        
        output = []
        for query in sql_queries:
            action = SNOWFLAKE_EXEC_SQL(query, is_save=False)
            output.append(action)
        
        column_info = algorithm_one(output)
        logger.info(f"Generated column info specification: {column_info}")
        return column_info
    
    def _prompt_agent_until_sql_query(self, obs):
        action = None
        last_action = None
        while not isinstance(action, SNOWFLAKE_EXEC_SQL):
            # Get action and execute until it's a SQL query
            _, action = self.predict(obs)

            if action is None:
                logger.info("Failed to parse action from response, try again.")
                # retry_count += 1
                # if retry_count > 3:
                #     logger.info("Failed to parse action from response, stop.")
                #     break
                obs = "Failed to parse action from your response, make sure you provide a valid action."
            else:
                if last_action is not None and last_action == action:
                    if repeat_action:
                        return False, "ERROR: Repeated action"
                    else:
                        obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action."
                        repeat_action = True
                else:
                    obs, _ = self.env.step(action)
                    last_action = action
                    repeat_action = False
        
        return action, obs
    
    def _validate_format(self, df_csv, format_spec):
        """
        Validates if a DataFrame follows the format specification.
        
        Parameters:
        -----------
        df_csv : pandas.DataFrame
            The DataFrame to validate
        format_spec : pandas.DataFrame
            A dataframe with columns representing the column names and their values representing their types, as well as a
            num_rows column with its value being the number of rows in the answer (-1 for any)
            
        Returns:
        --------
        bool
            True if the DataFrame follows the format specification, False otherwise
        """
        import pandas as pd
        import numpy as np
        
        # Create copies with lowercase column names to standardize
        df_csv_lower = df_csv.copy()
        df_csv_lower.columns = [col.lower() for col in df_csv_lower.columns]
        
        format_spec_lower = format_spec.copy()
        format_spec_lower.columns = [col.lower() for col in format_spec_lower.columns]
        
        # Check if all required columns exist
        columns = list(format_spec_lower.columns)[:-1]  # Exclude the num_rows column
        if not set(columns).issubset(set(df_csv_lower.columns)):
            print('hiiii')
            return False
        
        # Check number of rows if specified (not -1)
        if format_spec_lower['num_rows'].item() != -1 and len(df_csv_lower) != format_spec_lower['num_rows'].item():
            print('hi')
            return False
        
        # Check column types
        for col_name in columns:
            expected_type = format_spec_lower[col_name].item()  # Get the expected type from format_spec
            # Get the expected type string in lowercase for comparison
            expected_type_lower = str(expected_type).lower()
            
            # Handle different type specifications
            if expected_type_lower in ['int', 'integer']:
                # Check if all values in the column are integers
                try:
                    if not all(isinstance(val, (int, np.int64)) or (isinstance(val, float) and val.is_integer())
                              for val in df_csv_lower[col_name] if pd.notna(val)):
                        print('hiya')
                        return False
                except:
                    print('hello')
                    return False
            elif expected_type_lower in ['float', 'decimal', 'number']:
                # Check if all values in the column are numeric
                try:
                    if not pd.to_numeric(df_csv_lower[col_name], errors='coerce').notna().all():
                        return False
                except:
                    return False
            elif expected_type_lower in ['str', 'string', 'text']:
                # Check if all values in the column can be converted to string
                try:
                    df_csv_lower[col_name].astype(str)
                except:
                    return False
            elif expected_type_lower in ['date', 'datetime']:
                # Check if all values in the column can be parsed as dates
                try:
                    pd.to_datetime(df_csv_lower[col_name], errors='raise')
                except:
                    return False
            elif expected_type_lower in ['bool', 'boolean']:
                # Check if all values in the column are boolean
                try:
                    if not all(isinstance(val, (bool, np.bool_)) for val in df_csv_lower[col_name] if pd.notna(val)):
                        return False
                except:
                    return False
            # Add more type checks as needed
        
        # All checks passed
        return True

    def _self_refinement(self, prompt: str, exploration_results: Dict, format_spec: Dict):
        """
        Execute the self-refinement workflow.
        This is just algo 2 from the paper, but basically it's just refining the result over and over.
        """
        # Can also somewhat base this off of PromptAGent as far as executing actions goes, even though it'll just be sql
        # A lot of the PromptAgent stuff is written so that it can be multi-modal databases but we're just sql for now.
        # TODO: algorithm two from the paper (including validate result based on format spec)
        # logger.info("TODO: algorithm two from the paper")
        assert self.env is not None, "Environment is not set."
        itercount = 0
        results_tables = []
        last_message_index = len(self.history_messages)
        while itercount < self.max_steps: # number of overall individual! tries to give it and total possible queries
            error_count = 0
            format_error = 0
            obs = f"{str(prompt)}\nThese are the results of exploring the schema: {exploration_results}"
            self.history_messages = self.history_messages[:last_message_index] # reset history for each iteration
            # prompt until you get a sql query
            action, obs = self._prompt_agent_until_sql_query(obs)
            # Once you have a SQL query, refine until works or too many consecutive errors
            while error_count < 3 and format_error < 3:
                # TODO abstract from swamy algo 1
                if ("Error" not in obs or "Traceback" not in obs) and obs != "SQL command executed successfully. No output.":
                    
                    # Parse results into a DataFrame
                    if action.save_path:
                        df_csv = self._read_file_content(action.save_path)
                    else:
                        try:
                            df_csv = pd.read_csv(obs)
                        except Exception:
                            error_count += 1
                            action, obs = self._prompt_agent_until_sql_query(obs)

                    # Round numeric columns to two decimal places
                    numeric_columns = df_csv.select_dtypes(include=[np.number]).columns
                    df_csv[numeric_columns] = df_csv[numeric_columns].round(2)

                    # Explode any nested values
                    backup_copy = df_csv.copy()
                    try: 
                        columns_to_explode = []
                        for series in df_csv.columns:
                            if df_csv[series].dtype == "object":
                                columns_to_explode.append(series)
                        df_csv = df_csv.explode(columns_to_explode)
                    except ValueError:
                        df_csv = backup_copy
                        for series in df_csv.columns:
                            if df_csv[series].dtype == "object":
                                df_csv = df_csv.explode(series)

                    
                    if self._validate_format(df_csv, format_spec):
                        # Append to results if valid
                        normalized_df = [frozenset(row) for row in df_csv.values.tolist()]
                        results_tables.append(normalized_df)

                        # Check for self-consistency
                        if results_tables.count(normalized_df) >= 2:
                            logger.info("Self consistency satisfied")
                            return {tuple(normalized_df): action}
                        break # break out of loop for errors as we had a success
                    else:
                        format_error += 1
                        action, obs = self._prompt_agent_until_sql_query(f"The csv below you returned does not meet the required format.\n\nCSV Returned:\n {df_csv.to_dict()}\n\nRequired Format:\n{format_spec}\n\nPlease return a corrected query that returns results matching the required format.")
                else:
                    # Increment error counter
                    error_count += 1

                    # refine sql query and try again
                    action, obs = self._prompt_agent_until_sql_query(obs)
                    # If too many consecutive errors, terminate
                    if error_count >= 3:
                        logger.info(f"Max errors reach for iteration: {itercount}")
                        
            itercount += 1

        # Return final refined SQL and result
        return {results_tables[-1]: action} if results_tables and action else None
    
    def run(self):
        """
        Overall pipeline:
        1. Compress the schema.
        2. Generate format restrictions.
        3. Generate the initial prompt.
        4. Perform column exploration to clarify ambiguities.
        5. Execute self-refinement (in parallel threads) to generate the final SQL.
        6. Execute the final SQL and return the result.
        """
        # Step 0: Get the raw schema
        raw_schema = self._get_raw_schema()

        # Step 1: Compress schema
        compressed_schema = self._compress_schema(raw_schema)

        # Step 2: Generate format restrictions
        expected_format = self._generate_format_restrictions(compressed_schema)

        # Step 3: Generate initial prompt
        initial_prompt = self._generate_initial_prompt(compressed_schema, expected_format)

        # Step 4: Column exploration
        exploration_results = self._explore_columns(initial_prompt)
        # print(exploration_results)

        # Step 5: Self-refinement using parallel execution for robustness
        # TODO - fix with signals b/c they don't work if not in main thread :( big sad
        num_parallel_runs = 1
        # with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_runs) as executor:
        #     futures = [executor.submit(self._self_refinement,
        #                                  initial_prompt,
        #                                  exploration_results,
        #                                  expected_format)
        #                for _ in range(num_parallel_runs)]
        #     refined_sqls = [future.result() for future in futures]

        refined_results_to_sql = []
        for i in range(num_parallel_runs):
            refined_results_to_sql.append(self._self_refinement(initial_prompt, exploration_results, expected_format))
        refined_results_to_sql = [f for f in refined_results_to_sql if f is not None]
        results_list = [list(d.keys())[0] for d in refined_results_to_sql]
        # Simple voting mechanism: choose the SQL query that appears most frequently. TODO refine this maybe
        final_sql = refined_results_to_sql[0][results_list[0]]
        # final_sql = refined_results_to_sql[max(results_list, key=results_list.count)]
        logger.info(f"\nFinal SQL selected after voting:{final_sql}")

        # Step 6: Execute final query and return the result.
        self.env.step(SNOWFLAKE_EXEC_SQL(final_sql, is_save=True, save_path='results.csv'))
        final_obs = self.env.step(Terminate('results.csv'))
        return True, final_obs
    
        # TODO This does not include redoing if voting fails, or CTE-based refinement