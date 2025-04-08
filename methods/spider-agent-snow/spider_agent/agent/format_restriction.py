import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union

def generate_format(instruction: str, schema: Dict) -> Dict:
    """
    Generates expected answer format based on task instruction and schema.
    
    Args:
        instruction: Task instruction text.
        schema: Database schema information.
        
    Returns:
        Format specification for the expected answer.
    """
    format_spec = {
        'columns': [],
        'constraints': [],
        'example': {}
    }
    
    # Extract column information from the instruction
    format_spec['columns'] = _extract_columns_from_instruction(instruction, schema)
    
    # Extract constraints from the instruction
    format_spec['constraints'] = _extract_constraints(instruction)
    
    # Generate example data based on extracted columns and constraints
    format_spec['example'] = _generate_example_data(format_spec['columns'], format_spec['constraints'])
    
    return format_spec

def _extract_columns_from_instruction(instruction: str, schema: Dict) -> List[Dict]:
    """
    Extracts required columns from the instruction based on schema information.
    
    Args:
        instruction: Task instruction text.
        schema: Database schema information.
        
    Returns:
        List of column specifications including name, type, and description.
    """
    # Initialize columns list
    columns = []
    
    # Common patterns for column requests in instructions
    column_patterns = [
        r'(?:return|show|list|display|provide|find|get|identify|output)\s+(?:the\s+)?(.+?)(?:\s+from|\s+for|\s+of|\s+with|\s+where|\s+that|\s+when|\s+in|\s+\.|$)',
        r'(?:what\s+(?:is|are)\s+the\s+)(.+?)(?:\s+for|\s+of|\s+with|\s+where|\s+that|\s+when|\s+in|\s+\?|\s+\.|\?|\.|$)',
    ]
    
    # Look for column requests in the instruction
    potential_columns = []
    for pattern in column_patterns:
        matches = re.findall(pattern, instruction.lower())
        potential_columns.extend(matches)
    
    # Clean up extracted column text
    cleaned_columns = []
    for col_text in potential_columns:
        # Split by commas or "and" for multi-column requests
        parts = re.split(r'\s+and\s+|\s*,\s*', col_text)
        cleaned_columns.extend([part.strip() for part in parts if part.strip()])
    
    # Match potential columns with schema information
    for col_name in cleaned_columns:
        # Skip common non-column words
        if col_name in ['all', 'information', 'data', 'details', 'results']:
            continue
            
        # Search for matching columns in schema
        matched_col = _find_matching_column(col_name, schema)
        
        if matched_col:
            columns.append(matched_col)
        else:
            # If no match found, add as a derived column
            columns.append({
                'name': col_name,
                'type': 'unknown',
                'description': 'Derived from query',
                'is_derived': True
            })
    
    # If no columns found, default to returning all relevant columns
    if not columns:
        columns.append({
            'name': 'result',
            'type': 'any',
            'description': 'Query result',
            'is_derived': True
        })
    
    return columns

def _find_matching_column(col_name: str, schema: Dict) -> Optional[Dict]:
    """
    Finds a matching column in the schema based on name similarity.
    
    Args:
        col_name: Potential column name from instruction.
        schema: Database schema information.
        
    Returns:
        Column information if found, None otherwise.
    """
    # Implementation would depend on exact schema structure
    # This is a simplified version
    
    best_match = None
    highest_similarity = 0
    
    # For each table in schema
    for table_name, table_info in schema.items():
        if not isinstance(table_info, dict) or 'columns' not in table_info:
            continue
            
        # For each column in the table
        for column in table_info['columns']:
            column_name = column.get('name', '')
            
            # Calculate similarity between column name and requested name
            similarity = _calculate_name_similarity(col_name, column_name)
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = {
                    'name': column_name,
                    'type': column.get('type', 'unknown'),
                    'description': column.get('description', ''),
                    'table': table_name
                }
    
    # Return best match if similarity is above threshold
    if highest_similarity > 0.7:
        return best_match
    
    return None

def _calculate_name_similarity(name1: str, name2: str) -> float:
    """
    Calculates similarity between two column names.
    
    Args:
        name1: First name to compare.
        name2: Second name to compare.
        
    Returns:
        Similarity score between 0 and 1.
    """
    # Convert to lowercase and remove underscores for comparison
    name1_clean = name1.lower().replace('_', ' ')
    name2_clean = name2.lower().replace('_', ' ')
    
    # Check if one is contained in the other
    if name1_clean in name2_clean or name2_clean in name1_clean:
        return 0.8
    
    # Count matching words
    words1 = set(name1_clean.split())
    words2 = set(name2_clean.split())
    
    matching_words = words1.intersection(words2)
    
    if not words1 or not words2:
        return 0
    
    # Calculate Jaccard similarity
    return len(matching_words) / len(words1.union(words2))

def _extract_constraints(instruction: str) -> List[Dict]:
    """
    Extracts constraints from the instruction text.
    
    Args:
        instruction: Task instruction text.
        
    Returns:
        List of constraint specifications.
    """
    constraints = []
    
    # Check for count/limit constraints
    count_patterns = [
        r'(?:top|first|highest|lowest|best|worst)\s+(\d+)',
        r'limit\s+(?:to\s+)?(\d+)',
        r'(\d+)\s+(?:rows|results|records)',
    ]
    
    for pattern in count_patterns:
        matches = re.findall(pattern, instruction.lower())
        if matches:
            constraints.append({
                'type': 'limit',
                'value': int(matches[0])
            })
            break
    
    # Check for ordering constraints
    ordering_patterns = [
        r'(?:order|sort)(?:ed)?\s+by\s+(.+?)(?:\s+in\s+)?(ascending|descending|asc|desc)?(?:\s+order)?(?:\s+|\.|$)',
        r'(?:highest|lowest|maximum|minimum|max|min)\s+(.+?)(?:\s+|\.|$)',
    ]
    
    for pattern in ordering_patterns:
        matches = re.findall(pattern, instruction.lower())
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    col, direction = match
                else:
                    col, direction = match, None
                    
                # Determine direction based on keywords
                if direction in ['descending', 'desc'] or 'highest' in instruction.lower() or 'maximum' in instruction.lower() or 'max' in instruction.lower():
                    direction = 'desc'
                elif direction in ['ascending', 'asc'] or 'lowest' in instruction.lower() or 'minimum' in instruction.lower() or 'min' in instruction.lower():
                    direction = 'asc'
                else:
                    direction = 'asc'  # Default
                
                constraints.append({
                    'type': 'order',
                    'column': col.strip(),
                    'direction': direction
                })
    
    # Check for single-row constraint
    single_row_patterns = [
        r'(?:single|one|1)\s+(?:row|record|result)',
        r'(?:which|what)\s+(?:is|was|are|were)',
    ]
    
    for pattern in single_row_patterns:
        if re.search(pattern, instruction.lower()):
            constraints.append({
                'type': 'single_row',
                'value': True
            })
            break
            
    # Check for explicit format instructions
    format_patterns = [
        r'(?:return|show|list|display|provide|find|get|identify|output)\s+(?:the\s+)?results?\s+(?:in|as)\s+(.+?)(?:\s+|\.|$)',
        r'(?:format)\s+(?:the\s+)?results?\s+(?:in|as)\s+(.+?)(?:\s+|\.|$)',
    ]
    
    for pattern in format_patterns:
        matches = re.findall(pattern, instruction.lower())
        if matches:
            format_type = matches[0].strip()
            constraints.append({
                'type': 'format',
                'value': format_type
            })
            break
            
    # Check for specific attention instructions
    attention_patterns = [
        r'\(attention:?\s+(.+?)\)',
        r'attention:?\s+(.+?)(?:\.|$)',
    ]
    
    for pattern in attention_patterns:
        matches = re.findall(pattern, instruction, re.IGNORECASE)
        if matches:
            for attention_text in matches:
                constraints.append({
                    'type': 'attention',
                    'value': attention_text.strip()
                })
    
    return constraints

def _generate_example_data(columns: List[Dict], constraints: List[Dict]) -> Dict:
    """
    Generates example data based on column specifications and constraints.
    
    Args:
        columns: List of column specifications.
        constraints: List of constraint specifications.
        
    Returns:
        Dictionary containing example data structure.
    """
    example = {
        'header': [col['name'] for col in columns],
        'rows': []
    }
    
    # Determine number of example rows based on constraints
    num_rows = 1  # Default
    
    for constraint in constraints:
        if constraint['type'] == 'limit':
            num_rows = min(constraint['value'], 3)  # Cap at 3 for example
        elif constraint['type'] == 'single_row':
            num_rows = 1
            break
            
    # Generate placeholder example data
    for i in range(num_rows):
        row = []
        for col in columns:
            # Generate appropriate placeholder based on column type
            if col.get('type') == 'string' or col.get('type') == 'text':
                row.append(f"example_{col['name']}")
            elif col.get('type') == 'number' or col.get('type') == 'integer' or col.get('type') == 'float':
                row.append(i + 1)
            elif col.get('type') == 'date':
                row.append("YYYY-MM-DD")
            elif col.get('type') == 'timestamp':
                row.append("YYYY-MM-DD HH:MM:SS")
            else:
                row.append(f"value_{i+1}")
        
        example['rows'].append(row)
    
    return example

def validate_format(result: Any, expected_format: Dict) -> Tuple[bool, Optional[pd.DataFrame]]:
    """
    Validates that a query result matches the expected format.
    
    Args:
        result: Query execution result.
        expected_format: Expected format specification.
        
    Returns:
        Boolean indicating if the format is valid, and formatted result.
    """
    # Convert result to DataFrame if not already
    if isinstance(result, pd.DataFrame):
        df = result.copy()
    elif isinstance(result, dict):
        df = pd.DataFrame([result])
    elif isinstance(result, list):
        df = pd.DataFrame(result)
    elif isinstance(result, str):
        try:
            # Try to parse as CSV
            df = pd.read_csv(result)
        except:
            # If not CSV, try to convert to a single-cell DataFrame
            df = pd.DataFrame([[result]], columns=['result'])
    else:
        # Unknown format
        return False, None
    
    # Check if DataFrame is empty
    if df.empty:
        return False, None
        
    # Validate column requirements
    required_columns = [col['name'] for col in expected_format.get('columns', [])]
    
    # If no specific columns required, any non-empty result is valid
    if not required_columns:
        return True, df
    
    # Check if required columns exist in result (case insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    missing_columns = [col for col in required_columns if col.lower() not in df_columns_lower]
    
    # If required columns are missing, try to rename columns or derive missing ones
    if missing_columns:
        # Check if DataFrame has enough columns but with different names
        if len(df.columns) >= len(required_columns):
            # Try to map DataFrame columns to required columns
            column_mapping = {}
            for req_col in required_columns:
                for df_col in df.columns:
                    if _calculate_name_similarity(req_col, df_col) > 0.7:
                        column_mapping[df_col] = req_col
                        break
            
            # Rename columns based on mapping
            if column_mapping:
                df = df.rename(columns=column_mapping)
        
        # Check if we still have missing columns
        df_columns_lower = [col.lower() for col in df.columns]
        missing_columns = [col for col in required_columns if col.lower() not in df_columns_lower]
        
        # If still missing columns, try to derive them based on other columns
        if missing_columns:
            # This would be implemented based on specific application needs
            return False, None # TODO: look at this
    
    # Check count/limit constraints
    for constraint in expected_format.get('constraints', []):
        if constraint['type'] == 'limit':
            if len(df) > constraint['value']:
                df = df.head(constraint['value'])
                
        elif constraint['type'] == 'single_row' and constraint['value']:
            if len(df) > 1:
                df = df.head(1)
                
        elif constraint['type'] == 'order':
            col_name = constraint['column']
            direction = constraint['direction']
            
            # Find matching column in DataFrame
            matching_col = None
            for df_col in df.columns:
                if _calculate_name_similarity(col_name, df_col) > 0.7:
                    matching_col = df_col
                    break
                    
            if matching_col and matching_col in df.columns:
                ascending = (direction != 'desc')
                df = df.sort_values(by=matching_col, ascending=ascending)
    
    # At this point, the DataFrame should be valid or have been adjusted to match the format
    return True, df