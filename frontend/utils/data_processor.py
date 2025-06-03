import pandas as pd
import json
import os
import time
import random
import numpy as np
from typing import Dict, Any, List, Union
import tempfile
import logging
import sys

# Set up logging
temp_dir = (
    os.environ.get("TMPDIR")
    or os.environ.get("TEMP")
    or os.environ.get("TMP")
    or tempfile.gettempdir()
)
log_file = os.path.join(temp_dir, "data_processor.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(log_file, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Handle pandas Timestamp objects
        if hasattr(obj, "isoformat"):
            try:
                return obj.isoformat()
            except:
                return str(obj)
        # Handle pandas NaT (Not a Time)
        if pd.isna(obj):
            return None
        # Handle pandas Series
        if isinstance(obj, pd.Series):
            return obj.tolist()
        # Handle other pandas objects that might have a to_dict method
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except:
                return str(obj)
        # Handle datetime objects
        if hasattr(obj, "strftime"):
            try:
                return obj.isoformat()
            except:
                return str(obj)
        # Fallback for any other object types
        try:
            return str(obj)
        except:
            return super(NpEncoder, self).default(obj)


def main(args: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point for data processing operations"""
    try:
        logger.info(f"Starting data processing with args: {args}")
        operation = args.get("operation", "load")
        logger.info(f"Operation: {operation}")

        if operation == "load":
            result = load_data(args)
            logger.info(f"Load operation completed successfully")
            return result
        elif operation == "manipulate":
            result = manipulate_data(args)
            logger.info(f"Manipulate operation completed successfully")
            return result
        elif operation == "preview":
            result = get_data_preview(args)
            logger.info(f"Preview operation completed successfully")
            return result
        else:
            error_msg = f"Unknown operation: {operation}"
            logger.error(error_msg)
            return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error in main(): {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def load_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Load and analyze data file"""
    try:
        logger.info("Starting load_data operation")
        file_path = args["file_path"]
        file_name = args["file_name"]
        logger.info(f"Loading file: {file_name} from path: {file_path}")

        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return {"error": error_msg}

        # Check file size
        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")

        # Load data based on file extension
        if file_name.endswith(".csv"):
            logger.info("Loading CSV file")
            df = pd.read_csv(file_path)
        elif file_name.endswith((".xlsx", ".xls")):
            logger.info("Loading Excel file")
            df = pd.read_excel(file_path)
        else:
            error_msg = "Unsupported file format. Please use CSV or Excel files."
            logger.error(error_msg)
            return {"error": error_msg}

        logger.info(f"Data loaded successfully. Shape: {df.shape}")

        # Validate data
        if df.empty:
            error_msg = "The uploaded file is empty."
            logger.error(error_msg)
            return {"error": error_msg}

        if len(df) > 100000:
            error_msg = (
                f"File too large. Contains {len(df)} rows, max allowed is 100,000."
            )
            logger.error(error_msg)
            return {
                "error": "File too large. Please use files with fewer than 100,000 rows."
            }

        # Generate session ID
        session_id = f"session_{int(time.time())}_{random.randint(1000, 9999)}"
        logger.info(f"Generated session ID: {session_id}")

        # Prepare session data
        logger.info("Preparing session data")
        session_data = {
            "original_data": df.to_json(orient="records"),
            "current_data": df.to_json(orient="records"),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "created_at": time.time(),
            "file_name": file_name,
        }

        # Store in temporary session storage
        session_path = os.path.join(temp_dir, f"session_{session_id}.json")
        logger.info(f"Storing session data at: {session_path}")

        with open(session_path, "w") as f:
            json.dump(session_data, f, cls=NpEncoder)

        logger.info("Session data stored successfully")

        # Generate preview data (first 100 rows)
        logger.info("Generating preview data")
        preview_df = df.head(100).copy()

        # Convert any datetime columns to strings for JSON serialization
        for col in preview_df.columns:
            if preview_df[col].dtype == "datetime64[ns]" or "datetime" in str(
                preview_df[col].dtype
            ):
                preview_df[col] = preview_df[col].astype(str)

        preview_data = preview_df.to_dict("records")
        logger.info(f"Preview data generated with {len(preview_data)} rows")

        # Analyze data types and generate column info
        logger.info("Analyzing column information")
        column_info = []
        for col in df.columns:
            logger.debug(f"Analyzing column: {col}")
            col_data = {
                "name": col,
                "type": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": (
                    int(df[col].nunique())
                    if df[col].dtype != "object" or df[col].nunique() < 50
                    else None
                ),
            }

            # Add sample values for object columns
            if df[col].dtype == "object":
                col_data["sample_values"] = df[col].dropna().unique()[:10].tolist()

            column_info.append(col_data)

        result = {
            "session_id": session_id,
            "columns": column_info,
            "preview_data": preview_data,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "data_types": df.dtypes.astype(str).to_dict(),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
        }

        logger.info(f"Load operation completed successfully for session: {session_id}")
        return result

    except Exception as e:
        error_msg = f"Failed to load data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def manipulate_data(args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply data manipulation operations"""
    try:
        session_id = args["session_id"]
        operations = args["operations"]

        # Load session data
        session_path = os.path.join(temp_dir, f"session_{session_id}.json")
        if not os.path.exists(session_path):
            return {"error": "Session not found or expired"}

        with open(session_path, "r") as f:
            session_data = json.load(f)

        # Convert back to DataFrame
        import io

        df = pd.read_json(io.StringIO(session_data["current_data"]), orient="records")

        # Apply operations
        for operation in operations:
            op_type = operation.get("type")

            if op_type == "drop_columns":
                columns_to_drop = operation.get("columns", [])
                df = df.drop(columns=columns_to_drop, errors="ignore")

            elif op_type == "rename_columns":
                rename_mapping = operation.get("mapping", {})
                df = df.rename(columns=rename_mapping)

            elif op_type == "filter_data":
                # Apply date range filtering if specified
                date_column = operation.get("date_column")
                start_date = operation.get("start_date")
                end_date = operation.get("end_date")

                if date_column and date_column in df.columns:
                    # Convert to datetime
                    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

                    if start_date:
                        df = df[df[date_column] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df[date_column] <= pd.to_datetime(end_date)]

            elif op_type == "pivot_data":
                index_col = operation.get("index")
                columns_col = operation.get("columns")
                values_col = operation.get("values")

                if index_col and columns_col and values_col:
                    df = df.pivot_table(
                        index=index_col,
                        columns=columns_col,
                        values=values_col,
                        aggfunc="sum",
                    ).reset_index()

        # Update session data
        session_data["current_data"] = df.to_json(orient="records")
        session_data["columns"] = df.columns.tolist()
        session_data["dtypes"] = df.dtypes.astype(str).to_dict()

        # Save updated session
        with open(session_path, "w") as f:
            json.dump(session_data, f, cls=NpEncoder)

        # Generate new preview
        preview_df = df.head(100).copy()

        # Convert any datetime columns to strings for JSON serialization
        for col in preview_df.columns:
            if preview_df[col].dtype == "datetime64[ns]" or "datetime" in str(
                preview_df[col].dtype
            ):
                preview_df[col] = preview_df[col].astype(str)

        preview_data = preview_df.to_dict("records")

        return {
            "success": True,
            "preview_data": preview_data,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": df.columns.tolist(),
        }

    except Exception as e:
        return {"error": f"Failed to manipulate data: {str(e)}"}


def get_data_preview(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get data preview for a session"""
    try:
        logger.info("Starting get_data_preview operation")
        session_id = args["session_id"]
        logger.info(f"Requested session ID: {session_id}")

        # Load session data
        session_path = os.path.join(temp_dir, f"session_{session_id}.json")
        logger.info(f"Looking for session file at: {session_path}")

        if not os.path.exists(session_path):
            error_msg = (
                f"Session not found or expired. File does not exist: {session_path}"
            )
            logger.error(error_msg)

            # List all session files for debugging
            try:
                session_files = [
                    f
                    for f in os.listdir(temp_dir)
                    if f.startswith("session_") and f.endswith(".json")
                ]
                logger.info(f"Available session files: {session_files}")
                logger.info(f"Temp directory: {temp_dir}")
                logger.info(f"Looking for session file: session_{session_id}.json")
            except Exception as list_error:
                logger.error(f"Could not list session files: {list_error}")
                logger.error(f"Temp directory access failed: {temp_dir}")

            return {"error": f"Session not found or expired. Expected: {session_path}"}

        logger.info("Session file found, loading data")

        with open(session_path, "r") as f:
            session_data = json.load(f)

        logger.info(
            f"Session data loaded. File name: {session_data.get('file_name', 'Unknown')}"
        )
        logger.info(f"Session created at: {session_data.get('created_at', 'Unknown')}")

        # Convert back to DataFrame
        logger.info("Converting session data to DataFrame")
        try:
            import io

            df = pd.read_json(
                io.StringIO(session_data["current_data"]), orient="records"
            )
            logger.info(f"DataFrame created successfully. Shape: {df.shape}")
        except Exception as df_error:
            error_msg = f"Failed to convert session data to DataFrame: {str(df_error)}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}

        # Generate preview data
        logger.info("Generating preview data")
        preview_df = df.head(100).copy()

        # Convert any datetime columns to strings for JSON serialization
        for col in preview_df.columns:
            if preview_df[col].dtype == "datetime64[ns]" or "datetime" in str(
                preview_df[col].dtype
            ):
                preview_df[col] = preview_df[col].astype(str)

        preview_data = preview_df.to_dict("records")
        logger.info(f"Preview data generated with {len(preview_data)} rows")

        # Generate column info
        logger.info("Generating column information")
        column_info = []
        for col in df.columns:
            logger.debug(f"Processing column: {col}")
            col_data = {
                "name": col,
                "type": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
            }
            column_info.append(col_data)

        result = {
            "success": True,
            "preview_data": preview_data,
            "columns": column_info,
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "file_name": session_data.get("file_name", "Unknown"),
        }

        logger.info(
            f"Preview operation completed successfully for session: {session_id}"
        )
        logger.info(
            f"Returning {len(preview_data)} preview rows and {len(column_info)} columns"
        )

        return result

    except Exception as e:
        error_msg = f"Failed to get preview: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def cleanup_old_sessions():
    """Clean up old session files (older than 1 hour)"""
    try:
        current_time = time.time()
        # Use the same temp_dir as defined globally
        global temp_dir

        for filename in os.listdir(temp_dir):
            if filename.startswith("session_") and filename.endswith(".json"):
                file_path = os.path.join(temp_dir, filename)

                try:
                    with open(file_path, "r") as f:
                        session_data = json.load(f)

                    created_at = session_data.get("created_at", 0)

                    # Remove sessions older than 1 hour
                    if current_time - created_at > 3600:
                        os.remove(file_path)

                except (json.JSONDecodeError, KeyError, OSError):
                    # Remove corrupted session files
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

    except Exception:
        # Silent fail for cleanup
        pass


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Read arguments from file (new approach to fix Windows issues)
        args_file = sys.argv[1]
        try:
            logger.info(f"Reading arguments from file: {args_file}")
            with open(args_file, "r") as f:
                args = json.load(f)
            logger.info(f"Arguments loaded successfully: {args}")
            result = main(args)
            print(json.dumps(result, cls=NpEncoder))
        except Exception as e:
            error_msg = f"Failed to process arguments: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(json.dumps({"error": error_msg}, cls=NpEncoder))
    else:
        error_msg = "No arguments file provided"
        logger.error(error_msg)
        print(json.dumps({"error": error_msg}, cls=NpEncoder))
