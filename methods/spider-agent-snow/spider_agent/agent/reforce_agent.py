from spider_agent.agent.agents import PromptAgent
from spider_agent.envs.table_compression import compress_tables, compress_ddl_files
from spider_agent.agent.format_restriction import generate_format, validate_format
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
        from spider_agent.agent.action import Bash
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
    
    def _explore_columns(self) -> Dict:
        """
        Explore potentially useful columns through iterative SQL queries.
        
        Returns:
            Dict: Column exploration information
        """
        # Implementation of column exploration
        # This would be a more complex implementation in practice
        column_info = {"explored_columns": {}}
        
        # Generate and execute sample queries to understand the data
        sample_queries = self._generate_sample_queries()
        
        for query in sample_queries:
            try:
                from spider_agent.agent.action import SNOWFLAKE_EXEC_SQL
                action = SNOWFLAKE_EXEC_SQL(
                    sql_query=query,
                    is_save=False
                )
                
                obs, _ = self.env.step(action)
                
                # Parse and store the results
                parsed_results = self._parse_query_results(query, obs)
                if parsed_results:
                    table_name = self._extract_table_from_query(query)
                    if table_name:
                        column_info["explored_columns"][table_name] = parsed_results
            except Exception as e:
                logger.error(f"Error exploring column with query {query}: {str(e)}")
        
        return column_info
    
    def _generate_sample_queries(self) -> List[str]:
        """
        Generate sample SQL queries for column exploration.
        
        Returns:
            List[str]: List of SQL queries
        """
        # This would be generated by the LLM in practice
        # Here's a simplified placeholder
        queries = []
        
        # Get table names from the compressed schema
        table_names = list(self.compressed_schema.keys())
        
        for table_name in table_names[:5]:  # Limit to 5 tables for simplicity
            queries.append(f'SELECT * FROM {table_name} LIMIT 5')
        
        return queries
    
    def _parse_query_results(self, query: str, results: str) -> Dict:
        """
        Parse the results of a query execution.
        
        Args:
            query: SQL query
            results: Query execution results
            
        Returns:
            Dict: Parsed results
        """
        # Simple parsing for illustration
        lines = results.strip().split('\n')
        if len(lines) < 2:
            return None
            
        headers = lines[0].split('|')
        headers = [h.strip() for h in headers]
        
        data = []
        for line in lines[1:]:
            if '|' in line:
                values = line.split('|')
                values = [v.strip() for v in values]
                data.append(dict(zip(headers, values)))
        
        return {"headers": headers, "data": data}
    
    def _extract_table_from_query(self, query: str) -> Optional[str]:
        """
        Extract table name from a SQL query.
        
        Args:
            query: SQL query
            
        Returns:
            str: Table name or None
        """
        match = re.search(r'FROM\s+([^\s;]+)', query, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _execute_self_refinement(self, column_info: Dict) -> Tuple[bool, str]:
        """
        Execute the self-refinement workflow.
        
        Args:
            column_info: Column exploration information
            
        Returns:
            Tuple[bool, str]: (Success flag, Result)
        """
        iteration_count = 0
        results_tables = []
        
        while iteration_count < self.max_refinement_iterations:
            # Generate SQL query
            sql_query = self._generate_refined_sql(column_info, results_tables, iteration_count)
            
            # Execute the query
            try:
                from spider_agent.agent.action import SNOWFLAKE_EXEC_SQL
                action = SNOWFLAKE_EXEC_SQL(
                    sql_query=sql_query,
                    is_save=True,
                    save_path="/workspace/refined_result.csv"
                )
                
                obs, _ = self.env.step(action)
                
                # Check if results are valid and match the expected format
                if "saved to" in obs and not "No data found" in obs:
                    # Read the results file
                    result_content = self._read_results_file("/workspace/refined_result.csv")
                    
                    # Validate format
                    is_valid, formatted_result = validate_format(result_content, self.format_spec)
                    
                    if is_valid:
                        results_tables.append(formatted_result)
                        
                        # Check for self-consistency (same result appearing twice)
                        if self._is_self_consistent(results_tables):
                            from spider_agent.agent.action import Terminate
                            return True, "refined_result.csv"
            except Exception as e:
                logger.error(f"Error in self-refinement iteration {iteration_count}: {str(e)}")
            
            iteration_count += 1
        
        return False, ""
    
    def _generate_refined_sql(self, column_info: Dict, results_tables: List, iteration: int) -> str:
        """
        Generate a refined SQL query based on previous results.
        
        Args:
            column_info: Column exploration information
            results_tables: Previous query results
            iteration: Current iteration number
            
        Returns:
            str: Refined SQL query
        """
        # In practice, this would involve calling the LLM
        # Here's a simplified placeholder
        if not results_tables:
            # First iteration - generate initial query
            return "SELECT * FROM example_table LIMIT 10"
        else:
            # Subsequent iterations - refine based on previous results
            return f"SELECT * FROM example_table WHERE some_column > {iteration} LIMIT 10"
    
    def _read_results_file(self, file_path: str):
        """
        Read content from a results file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content of the file
        """
        from spider_agent.agent.action import Bash
        obs, _ = self.env.step(Bash(code=f"cat {file_path}"))
        
        try:
            # Try to parse as CSV
            lines = obs.strip().split('\n')
            if lines:
                headers = lines[0].split(',')
                data = []
                for line in lines[1:]:
                    data.append(dict(zip(headers, line.split(','))))
                return {"headers": headers, "data": data}
        except Exception:
            pass
            
        return obs
    
    def _is_self_consistent(self, results_tables: List) -> bool:
        """
        Check if results are self-consistent (same result appears twice).
        
        Args:
            results_tables: List of query results
            
        Returns:
            bool: True if self-consistent, False otherwise
        """
        # Simplified implementation - actual implementation would be more robust
        if len(results_tables) < 2:
            return False
            
        # Compare string representations of the last two results
        return str(results_tables[-1]) == str(results_tables[-2])
    
    def _execute_cte_refinement(self) -> Tuple[bool, str]:
        """
        Execute CTE-based refinement for unresolved cases.
        
        Returns:
            Tuple[bool, str]: (Success flag, Result)
        """
        # Implementation of CTE-based refinement
        # This is a simplified placeholder
        try:
            # Generate a CTE-based SQL query
            cte_query = """
            WITH first_cte AS (
                SELECT * FROM example_table WHERE column1 > 0
            ),
            second_cte AS (
                SELECT * FROM first_cte WHERE column2 < 100
            )
            SELECT * FROM second_cte LIMIT 10
            """
            
            from spider_agent.agent.action import SNOWFLAKE_EXEC_SQL
            action = SNOWFLAKE_EXEC_SQL(
                sql_query=cte_query,
                is_save=True,
                save_path="/workspace/cte_result.csv"
            )
            
            obs, _ = self.env.step(action)
            
            if "saved to" in obs and not "No data found" in obs:
                from spider_agent.agent.action import Terminate
                return True, "cte_result.csv"
        except Exception as e:
            logger.error(f"Error in CTE refinement: {str(e)}")
        
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