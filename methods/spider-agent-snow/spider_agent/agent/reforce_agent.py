from spider_agent.agent.agents import PromptAgent
from spider_agent.envs.table_compression import compress_tables, compress_ddl_files

from typing import Dict, List, Any, Optional, Tuple
import threading
import pandas as pd
import logging
import concurrent.futures
import json
import re

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
        
        # Apply table compression to the schema information
        self.compressed_schema = self._compress_schema()
        
        # Generate expected format based on instruction and schema
        self.format_spec = self._generate_format_restrictions()


    
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
        This is done by giving a separate chat session an format prompt.
        It then gives you a csv or smth i don't know that yet with the necessary columns and types and stuff.
        That will be used/enforced later in self-refinement.
        
        Returns:
            Dict: Format specification
        """
        # This is just ripped out of the paper describing how the format spec needs to be.
        # it wasn't actually said to be their prompt but it sure reads like it
        # TODO - definitely needs some editing on this prompt
        format_prompt =  (
            "The response must strictly adhere to the specified format in CSV style, ensuring alignment with executed CSV files. "
            "Each column should be explicitly defined, including all necessary attributes, and each record should occupy a separate row. "
            "The format should account for specific cases, such as superlatives, percentages, or coordinates, ensuring the output is concise, clear, and unambiguous. "
            "For ambiguous terms, potential values or additional columns can be added to maintain clarity and precision."
        )
        # If this ^^^ prompt isn't working, we might need to make a separate system prompt
        # TODO fix this llm call (use history messages and such)
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
        # If this ^^^ prompt isn't working, we might need to make a separate system prompt
        output = call_llm(exploration_prompt, self.model, self.max_tokens, self.temperature, self.top_p)
        # TODO - Parse the output into list of sql queries (similar to predict from PromptAgent)
        # TODO - Algorithm one from the paper
        def algorithm_one(output):
            # TODO
            return "column info" 
        
        column_info = algorithm_one(output)

        logger.info(f"Generated column info specification: {column_info}")
        
        return column_info
    
    def _self_refinement(self, prompt: str, exploration_results: Dict, format_spec: Dict, max_iter=3):
        """
        Execute the self-refinement workflow.
        This is just algo 2 from the paper, but basically it's just refining the result over and over.
        """
        # Can also somewhat base this off of PromptAGent as far as executing actions goes, even though it'll just be sql
        # A lot of the PromptAgent stuff is written so that it can be multi-modal databases but we're just sql for now.
        # TODO: algorithm two from the paper (including validate result based on format spec)
        logger.info("TODO: algorithm two from the paper")
        pass
    
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
        final_result = self.db.execute(final_sql)
        return final_result
    
        # TODO This does not include redoing if voting fails, or CTE-based refinement