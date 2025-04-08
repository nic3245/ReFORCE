from spider_agent.agent.agents import PromptAgent
from spider_agent.envs.table_compression import compress_tables, compress_ddl_files
from spider_agent.agent.format_restriction import generate_format, validate_format
from spider_agent.agent.column_exploration import explore_columns
from spider_agent.agent.self_refinement import self_refinement_workflow, cte_based_refinement

from typing import Dict, List, Any, Optional, Tuple
import threading
import pandas as pd
import logging
import concurrent.futures
import json
import re

from spider_agent.agent.prompts import SNOWFLAKE_REFORCE_SYSTEM
from spider_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS


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
        self.format_spec = self._generate_format()
        
    def _compress_schema(self) -> Dict:
        """
        Apply table compression to the database schema to reduce context size.
        
        Returns:
            Dict: Compressed schema information
        """
        # Extract schema information from the environment
        schema_info = self._extract_schema_info()
        
        # Apply compression function
        compressed_schema = compress_tables(schema_info)
        
        logger.info(f"Schema compressed from {len(str(schema_info))} to {len(str(compressed_schema))} characters")
        
        return compressed_schema
    
    def _extract_schema_info(self) -> Dict:
        """
        Extract schema information from the environment.
        
        Returns:
            Dict: Schema information
        """
        # This implementation would depend on how schema is stored in the environment
        # For Spider-Agent, we'll need to parse from files
        schema_info = {}
        
        # Execute bash commands to gather schema information
        obs, _ = self.env.step(self._create_schema_exploration_action())
        
        # Parse the observation to extract schema info
        # This is a simplified version - actual implementation would be more complex
        for line in obs.split('\n'):
            if '.json' in line or '.csv' in line:
                file_path = line.strip()
                content = self._read_file_content(file_path)
                if content:
                    schema_info[file_path] = content
        
        return schema_info
    
    def _create_schema_exploration_action(self):
        """
        Create an action to explore the schema.
        
        Returns:
            Action: Bash action to list schema files
        """
        return Bash(code="find . -name '*.json' -o -name '*.csv' | grep -v 'result.json' | grep -v 'results.csv'")
    
    def _read_file_content(self, file_path: str):
        """
        Read content from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content of the file or None if error
        """
        obs, _ = self.env.step(Bash(code=f"cat {file_path}"))
        
        try:
            if file_path.endswith('.json'):
                return json.loads(obs)
            elif file_path.endswith('.csv'):
                # Simple CSV parsing for illustration
                lines = obs.strip().split('\n')
                if lines:
                    headers = lines[0].split(',')
                    data = []
                    for line in lines[1:]:
                        data.append(dict(zip(headers, line.split(','))))
                    return {"headers": headers, "data": data}
            return obs
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def _generate_format(self) -> Dict:
        """
        Generate expected answer format based on task instruction and schema.
        
        Returns:
            Dict: Format specification
        """
        # Use the format restriction module to generate format
        format_spec = generate_format(self.instruction, self.compressed_schema)
        
        logger.info(f"Generated format specification: {format_spec}")
        
        return format_spec
    
    def _explore_columns(self):
        """
        Apply column exploration to the database schema.
        
        Returns:
            Dict: Column information
        """
        # Use the explore columns module to generate info
        column_info = explore_columns()
        
        logger.info(f"Generated column info specification: {column_info}")
        
        return column_info
    
    def _execute_self_refinement(column_info):
        """
        Execute the self-refinement workflow.
        """
        # Use the self-refinement module to generate SQL
        # TODO
        done, result = self_refinement_workflow()
        
        logger.info(f"Generated result from self-refinement: {done}, {result}")

        return done, result
    
    def _execute_cte_refinement():
        """
        Execute the CTE-based refinement workflow.
        """
        # Use the CTE-based refinement module to generate SQL
        # TODO
        done, result = cte_based_refinement()
        
        logger.info(f"Generated result from CTE-based refinement: {done}, {result}")

        return done, result
    
    def run(self):
        """
        Override the run method to implement the ReFoRCE workflow.
        """
        assert self.env is not None, "Environment is not set."
        
        # If parallel execution is enabled, run multiple threads
        if self.parallel_count > 1:
            return self._run_parallel()
        else:
            return self._run_single()
    
    def _run_single(self):
        """
        Run a single ReFoRCE workflow.
        
        Returns:
            Tuple[bool, str]: (Success flag, Result)
        """
        # 1. Apply column exploration
        column_info = self._explore_columns()
        
        # 2. Execute self-refinement workflow
        done, result = self._execute_self_refinement(column_info)
        
        # 3. If no result, try CTE-based refinement
        if not done or not result:
            done, result = self._execute_cte_refinement()
        
        return done, result
    
    def _run_parallel(self):
        """
        Run multiple ReFoRCE workflows in parallel and aggregate results.
        
        Returns:
            Tuple[bool, str]: (Success flag, Result)
        """
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel_count) as executor:
            futures = [executor.submit(self._run_single) for _ in range(self.parallel_count)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    done, result = future.result()
                    if done and result:
                        results.append((done, result))
                except Exception as e:
                    logger.error(f"Error in parallel execution: {str(e)}")
        
        # If we have results, apply voting mechanism
        if results:
            return self._vote_results(results)
        else:
            return False, ""
    
    def _vote_results(self, results: List[Tuple[bool, str]]) -> Tuple[bool, str]:
        """
        Apply voting mechanism to determine the most likely correct outcome.
        
        Args:
            results: List of (done, result) tuples from parallel runs
            
        Returns:
            Tuple[bool, str]: (Success flag, Result) with the highest vote
        """
        # Count occurrences of each result
        result_counts = {}
        
        for done, result in results:
            if done and result:
                result_counts[result] = result_counts.get(result, 0) + 1
        
        if not result_counts:
            return False, ""
        
        # Find the result with the highest count
        best_result = max(result_counts.items(), key=lambda x: x[1])[0]
        
        return True, best_result