import re
import os
import json
from typing import Dict, List, Tuple, Set

def compress_tables(database_schema: Dict) -> Dict:
    """
    Compresses table information by pattern-based matching of similar tables.
    For example, tables like GA_SESSIONS_20160801 through GA_SESSIONS_20170801 
    would be compressed to a single representative table.
    
    Args:
        database_schema: Dictionary containing database schema information.
        
    Returns:
        Compressed database schema.
    """
    # Create a copy of the schema to avoid modifying the original
    compressed_schema = database_schema.copy()
    
    # Extract table names
    table_names = list(database_schema.keys()) if isinstance(database_schema, dict) else []
    
    # Find table groups with similar naming patterns
    table_groups = _identify_table_patterns(table_names)
    
    # Compress each group
    for pattern, tables in table_groups.items():
        if len(tables) <= 1:  # Skip if there's only one table or none
            continue
            
        # Choose a representative table (first in the list)
        representative_table = tables[0]
        
        # Keep only the representative in compressed schema
        for table in tables[1:]:
            if table in compressed_schema:
                del compressed_schema[table]
                
        # Optionally, add a note about compressed tables
        if representative_table in compressed_schema:
            if isinstance(compressed_schema[representative_table], dict):
                compressed_schema[representative_table]["compressed_tables"] = tables[1:]
            elif hasattr(compressed_schema[representative_table], "__dict__"):
                compressed_schema[representative_table].compressed_tables = tables[1:]
    
    return compressed_schema

def _identify_table_patterns(table_names: List[str]) -> Dict[str, List[str]]:
    """
    Identifies groups of tables that follow similar naming patterns.
    
    Args:
        table_names: List of table names to analyze.
        
    Returns:
        Dictionary mapping patterns to lists of tables that match each pattern.
    """
    # Dictionary to store pattern groups
    pattern_groups = {}
    
    # Common patterns in SQL databases, especially for time-based tables
    # 1. Tables with date suffixes: table_20220101, table_20220102, etc.
    date_suffix_pattern = r'(.+?)_(\d{8})$'
    
    # 2. Tables with year/month/day components: table_2022_01_01, table_2022_01_02, etc.
    date_components_pattern = r'(.+?)_(\d{4})_(\d{2})_(\d{2})$'
    
    # 3. Tables with version numbers: table_v1, table_v2, etc.
    version_pattern = r'(.+?)_v(\d+)$'
    
    # 4. Tables with partition indicators: table_p1, table_p2, etc.
    partition_pattern = r'(.+?)_p(\d+)$'
    
    # Process each table name
    for table_name in table_names:
        # Check for date suffix pattern
        match = re.match(date_suffix_pattern, table_name)
        if match:
            base_name = match.group(1)
            pattern_key = f"{base_name}_DATE"
            pattern_groups.setdefault(pattern_key, []).append(table_name)
            continue
            
        # Check for date components pattern
        match = re.match(date_components_pattern, table_name)
        if match:
            base_name = match.group(1)
            pattern_key = f"{base_name}_YEAR_MONTH_DAY"
            pattern_groups.setdefault(pattern_key, []).append(table_name)
            continue
            
        # Check for version pattern
        match = re.match(version_pattern, table_name)
        if match:
            base_name = match.group(1)
            pattern_key = f"{base_name}_VERSION"
            pattern_groups.setdefault(pattern_key, []).append(table_name)
            continue
            
        # Check for partition pattern
        match = re.match(partition_pattern, table_name)
        if match:
            base_name = match.group(1)
            pattern_key = f"{base_name}_PARTITION"
            pattern_groups.setdefault(pattern_key, []).append(table_name)
            continue
            
        # If no specific pattern is found, each table is its own group
        pattern_groups.setdefault(table_name, []).append(table_name)
    
    return pattern_groups

def compress_ddl_files(ddl_directory: str) -> Dict[str, str]:
    """
    Compresses DDL files by identifying similar tables and keeping only 
    representative examples.
    
    Args:
        ddl_directory: Path to directory containing DDL files.
        
    Returns:
        Dictionary mapping table patterns to representative DDL file paths.
    """
    # Get all DDL files in the directory
    ddl_files = []
    for root, _, files in os.walk(ddl_directory):
        for file in files:
            if file.endswith('.csv') or file.endswith('.json'):
                ddl_files.append(os.path.join(root, file))
    
    # Extract table names from filenames
    table_names = [os.path.splitext(os.path.basename(file))[0] for file in ddl_files]
    
    # Identify patterns
    table_groups = _identify_table_patterns(table_names)
    
    # Map patterns to representative DDL files
    compressed_ddls = {}
    for pattern, tables in table_groups.items():
        if len(tables) <= 1:  # Skip if there's only one table or none
            continue
            
        # Find the DDL file for the representative table
        representative_table = tables[0]
        for ddl_file in ddl_files:
            if os.path.splitext(os.path.basename(ddl_file))[0] == representative_table:
                compressed_ddls[pattern] = ddl_file
                break
    
    return compressed_ddls