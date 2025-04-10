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

from spider_agent.agent.prompts import SNOWFLAKE_REFORCE_SYSTEM
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
        self.parallel_count = parallel_count
        self.max_refinement_iterations = max_refinement_iterations
        self.results_cache = {}  # Cache for storing execution results
        self.format_specs = {}   # Store format specifications
        self.correction_messages = []
        
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
                # Simple CSV parsing for illustration TODO - REDO WITH CSV MODULE LOL (remember it is a string from cat tho)
                lines = obs.strip().split('\n')
                if lines:
                    headers = lines[0].split(',')
                    data = []
                    for line in lines[1:]:
                        data.append(dict(zip(headers, line.split(','))))
                    logger.info(f"Data from _read_file_content: {data}")
                    return {"headers": headers, "data": data}
            return obs
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
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

        logger.info(f"Schema extracted files: {schema_info.keys()[:min(5, len(schema_info.keys()))]}")
        
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
    
    def _generate_format_restrictions(self) -> Dict:
        """
        Generate expected answer format based on task instruction and schema. 
        This is done by giving a separate chat session and format prompt.
        It then gives you a csv or smth i don't know that yet with the necessary columns and types and stuff.
        That will be used/enforced later in self-refinement.
        
        Returns:
            Dict: Format specification
        """
        # This is just ripped out of the paper describing how the format spec needs to be.
        # it wasn't actually said to be their prompt but it sure reads like it
        # TODO - definitely needs some editing on this prompt
        # TODO - make a different system prompt and history messages for only formatting
        format_prompt =  (
            "The response must strictly adhere to the specified format in CSV style, ensuring alignment with executed CSV files. "
            "Each column should be explicitly defined, including all necessary attributes, and each record should occupy a separate row. "
            "The format should account for specific cases, such as superlatives, percentages, or coordinates, ensuring the output is concise, clear, and unambiguous. "
            "For ambiguous terms, potential values or additional columns can be added to maintain clarity and precision."
        )
        # If this ^^^ prompt isn't working, we might need to make a separate system prompt
        # TODO fix this llm call (use history messages and such instead of format_prompt only)
        # FIXME doesn't work, make it into a payload
        # XXX
        format_spec = call_llm(self.env, format_prompt)
        # TODO parse the output probably
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
    
    def _explore_columns(self, prompt: str) -> Dict:
        """
        Apply column exploration to the database schema.
        Column exploration is done by giving a separate chat session an exploration prompt and getting a list of sql queries.
        It then executes the queries and returns the results to the agent, which can either keep exploring or quit.
        Returns:
            Dict: Column information
        """
        exploration_prompt = f"""{prompt}
        ######## COLUMN EXPLORATION ########
        Before tackling the task, first generate SQL queries to carry out column exploration. Identify relevant columns and sample values that will help complete the task."""
        # If this ^^^ prompt isn't working, we might need to make a separate system prompt TODO make into system prompt with separate chat message
        output = call_llm(exploration_prompt, self.model, self.max_tokens, self.temperature, self.top_p)
        # TODO - Parse the output into list of sql queries (similar to predict from PromptAgent)
        # TODO - Algorithm one from the paper
        def algorithm_one(sql_actions):
            def self_correct(sql_repr, result):
                correction_prompt = f"""
                    You are an expert SQL assistant. You will be given:
                    - An incorrect user generated SQL query
                    - The results from running that query

                    Your task is to **correct the SQL query** so that it runs correctly and returns valid, structured results. 
                    Use all previous information provided to you and your expert SQL knowledge to correct the query. 

                    Here are your goals:
                    1. Fix all syntax errors in the SQL query.
                    2. Logically verify that the query can return the intended correct results.
                    3. Ensure the query returns a non-empty, meaningful result.
                    4. Avoid columns with all NULL values or obviously invalid data types.
                    5. If the original query seems too complex or ambiguous, simplify it while preserving intent.
                    
                    Respond with **only one corrected SQL query**. Do not include any explanation or comments.

                    User Generated SQL Query:
                    {sql_repr}

                    Execution Result:
                    {result}
                """

                header = ""
                correction = self._get_llm_response(correction_prompt, header, self.correction_messages)
                correction_result = f"""

                    The previous query was corrected as follows:

                    Corrected SQL Query:
                    {correction}

                    Please use this as context for future improvements or related queries.
                """

                message = correction_prompt + correction_result
                
                self.correction_messages.append({
                    "role": "user", # XXX idk if this works
                    "content": [
                        {
                            "type": "text",
                            "text": f"{message}"
                        }
                    ]
                })  
                
                return correction
            

            result_dic = {}
            error_rec = 0

            while len(sql_actions) > 0:
                sql_action = sql_actions.pop(0)
                result = self.env.step(sql_action)
                if ("error" not in result or "traceback" not in result) and result != "SQL command executed successfully. No output.":
                        df = pd.read_csv(StringIO(result))
                        empty_cols = df.columns[df.isnull().all()]
                        if len(empty_cols) == 0:
                            result_dic[sql_action] = df
                            error_rec = 0
                            message = f"""
                                The following SQL query executed successfully and returned correct results:

                                Query:
                                {sql_action.__repr__()}

                                Result (first 5 or fewer rows):
                                {df.head(5).to_string(index=False)}

                                This query does not need need to be revised. Please remember this query and its result as context for future corrections.
                                """
                            self.correction_messages.append({
                                "role": "user", # XXX idk if this works
                                "content": [
                                    {
                                        "type": "text",
                                        "text": f"{message}"
                                    }
                                ]
                            })  
                            continue

                # correction
                max_iter = 3
                simplify = False
                corrected_sql = None

                for i in range(max_iter):
                    # TODO - implement method add chat_session as parameter
                    sql_repr = sql_action.__repr__()
                    corrected_sql = self_correct(sql_repr, result)
                    result = self.env.step(corrected_sql)
                    if "error" not in result or "traceback" not in result:
                        if result != "SQL command executed successfully. No output.":
                            df = pd.read_csv(StringIO(result))
                            empty_cols = df.columns[df.isnull().all()]
                            if len(empty_cols) == 0:
                                result_dic[sql_action] = df
                                error_rec = 0

                                # apply correction to rest of sql_actions
                                for i in range(len(sql_actions)):
                                    next_sql_repr = sql_actions[i].__repr__()
                                    next_corrected_sql = self_correct(next_sql_repr, result)
                                    sql_actions[i] = next_corrected_sql
                                break

                error_rec += 1
                if error_rec > 5:
                    return result_dic
            return result_dic 
        
        column_info = algorithm_one(output)

        logger.info(f"Generated column info specification: {column_info}")
        
        return column_info
    
    def _prompt_agent_until_sql_query(self, obs):
        action = None
        while not isinstance(action, SNOWFLAKE_EXEC_SQL):
            # Get action and execute until it's a SQL query
            _, action = self.predict(obs)

            if action is None:
                logger.info("Failed to parse action from response, try again.")
                retry_count += 1
                if retry_count > 3:
                    logger.info("Failed to parse action from response, stop.")
                    break
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
    
    def _self_refinement(self, prompt: str, exploration_results: Dict, format_spec: Dict):
        """
        Execute the self-refinement workflow.
        This is just algo 2 from the paper, but basically it's just refining the result over and over.
        """
        # Can also somewhat base this off of PromptAGent as far as executing actions goes, even though it'll just be sql
        # A lot of the PromptAgent stuff is written so that it can be multi-modal databases but we're just sql for now.
        # TODO: algorithm two from the paper (including validate result based on format spec)
        logger.info("TODO: algorithm two from the paper")
        assert self.env is not None, "Environment is not set."
        itercount = 0
        results_tables = []
        last_message_index = len(self.history_messages)
        while itercount < self.max_steps: # number of overall individual! tries to give it and total possible queries
            error_count = 0
            obs = f"{str(prompt)}\nThese are the results of exploring the schema: {exploration_results}"
            self.history_messages = self.history_messages[:last_message_index] # reset history for each iteration
            # prompt until you get a sql query
            action, obs = self._prompt_agent_until_sql_query(obs)
            # Once you have a SQL query, refine until works or too many consecutive errors
            while error_count < 3:
                # TODO abstract from swamy algo 1
                if response_is_valid(obs):
                    # Parse results into a DataFrame (assume this function exists)
                    df_csv = parse_to_dataframe(response)

                    # Round numeric columns to two decimal places
                    df_csv = round_numeric_columns(df_csv)

                    # Check if there are nested values (assume this function exists)
                    if not has_nested_values(df_csv) and is_expected_format(df_csv):
                        # Append to results if valid
                        results_tables.append((action, df_csv))
                        
                        # Save results (assume these functions exist)
                        save_refined_query(action)
                        save_results(df_csv)

                        # Check for self-consistency
                        if results_tables.count((action, df_csv)) >= 2:
                            logger.info("Self consistency satisfied")
                            return results_tables
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
        return results_tables if results_tables else None
    
    def run(self, natural_language_query):
        """
        Overall pipeline:
        1. Compress the schema.
        2. Generate format restrictions.
        3. Generate the initial prompt.
        4. Perform column exploration to clarify ambiguities.
        5. Execute self-refinement (in parallel threads) to generate the final SQL.
        6s. Execute the final SQL and return the result.
        """
        # Step 0: Get the raw schema
        raw_schema = self._get_raw_schema()

        # Step 1: Compress schema
        compressed_schema = self._compress_schema(raw_schema)

        # Step 2: Generate format restrictions
        expected_format = self._generate_format_restrictions()

        # Step 3: Generate initial prompt
        initial_prompt = self._generate_initial_prompt(compressed_schema, expected_format, natural_language_query)

        # Step 4: Column exploration
        exploration_results = self._explore_columns(initial_prompt)

        # Step 5: Self-refinement using parallel execution for robustness
        num_parallel_runs = 3
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_runs) as executor:
            futures = [executor.submit(self._self_refinement,
                                         initial_prompt,
                                         exploration_results,
                                         expected_format)
                       for _ in range(num_parallel_runs)]
            refined_sqls = [future.result() for future in futures]

        # Simple voting mechanism: choose the SQL query that appears most frequently. TODO refine this maybe
        final_sql = max(set(refined_sqls), key=refined_sqls.count)
        logger.info(f"\nFinal SQL selected after voting:{final_sql}")

        # Step 6: Execute final query and return the result.
        self.env.step(SNOWFLAKE_EXEC_SQL(final_sql, is_save=True, save_path='results.csv'))
        final_obs = self.env.step(Terminate('results.csv'))
        return final_sql
    
        # TODO This does not include redoing if voting fails, or CTE-based refinement