import streamlit as st
import time
import logging

logger = logging.getLogger(__name__)

def initialize_batch_process_state():
    """
    Initializes the batch process manager in session state if it doesn't exist.
    """
    if 'batch_process_manager' not in st.session_state:
        st.session_state.batch_process_manager = {
            'is_active': False,
            'current_stage': None,
            'stage_config': {},  # To store any configuration specific to the current stage
            'files_to_process': [],
            'total_files': 0,
            'current_index': 0,
            'processed_in_sub_batch': 0, # How many files processed in the current sub-batch iteration
            'results': [],  # Stores dicts with {'file_id', 'name', 'status', 'message', 'data'}
            'error_count': 0,
            'successful_count': 0,
            'user_cancelled': False,
            'sub_batch_size': 1,          # Number of files to process before a potential short pause
            'api_call_delay_seconds': 0.5, # Delay between individual API calls within a sub-batch
            'sub_batch_delay_seconds': 0.1 # Delay after a sub-batch completes (if not the end of all files)
        }
        logger.info("Batch Process Manager initialized in session state.")

def start_new_batch(stage_name: str, files: list, stage_specific_config: dict = None,
                    sub_batch_size: int = 1, api_delay: float = 0.5, batch_delay: float = 0.1):
    """
    Starts a new batch processing operation.
    """
    initialize_batch_process_state() # Ensure it's initialized

    bpm = st.session_state.batch_process_manager
    bpm['is_active'] = True
    bpm['current_stage'] = stage_name
    bpm['stage_config'] = stage_specific_config if stage_specific_config is not None else {}
    bpm['files_to_process'] = list(files) # Ensure it's a list
    bpm['total_files'] = len(files)
    bpm['current_index'] = 0
    bpm['processed_in_sub_batch'] = 0
    bpm['results'] = []
    bpm['error_count'] = 0
    bpm['successful_count'] = 0
    bpm['user_cancelled'] = False
    bpm['sub_batch_size'] = sub_batch_size
    bpm['api_call_delay_seconds'] = api_delay
    bpm['sub_batch_delay_seconds'] = batch_delay

    logger.info(f"Starting new batch for stage '{stage_name}'. Total files: {bpm['total_files']}, Sub-batch size: {sub_batch_size}.")
    st.rerun()

def record_batch_file_result(file_id: str, name: str, status: str, message: str = "", data: dict = None):
    """
    Records the result of processing a single file in the batch.
    status should be 'success' or 'error'.
    """
    if 'batch_process_manager' not in st.session_state:
        logger.error("Attempted to record batch file result, but Batch Process Manager is not initialized.")
        return

    bpm = st.session_state.batch_process_manager
    bpm['results'].append({
        'file_id': file_id,
        'name': name,
        'status': status,
        'message': message,
        'data': data if data is not None else {}
    })

    if status.lower() == 'success':
        bpm['successful_count'] += 1
    else:
        bpm['error_count'] += 1

def cancel_current_batch():
    """
    Flags the current batch processing operation for cancellation.
    """
    if 'batch_process_manager' not in st.session_state:
        logger.warning("Attempted to cancel batch, but Batch Process Manager is not initialized.")
        return

    bpm = st.session_state.batch_process_manager
    if bpm['is_active']:
        bpm['user_cancelled'] = True
        logger.info(f"User initiated cancellation for stage '{bpm['current_stage']}'.")
    else:
        logger.info("Attempted to cancel, but no batch process is currently active.")

def run_batch_processing_iteration(stage_processor_fn):
    """
    Runs a single iteration of the batch processing loop.
    This function is designed to be called repeatedly (e.g., via st.rerun)
    until the batch is complete or cancelled.

    Args:
        stage_processor_fn: A function that takes (file_info: dict, stage_config: dict)
                            and processes a single file. It should return a dictionary
                            with {'status': 'success'/'error', 'message': '...', 'data': {...}}
                            or raise an exception.
    """
    if 'batch_process_manager' not in st.session_state:
        logger.warning("run_batch_processing_iteration called but BPM not initialized. Initializing now.")
        initialize_batch_process_state() # Should ideally be initialized before calling this

    bpm = st.session_state.batch_process_manager

    # Handle Inactive/Cancelled (Top)
    if not bpm['is_active'] or bpm['user_cancelled']:
        if bpm['user_cancelled'] and bpm['is_active']: # Ensure we only log and rerun once for cancellation
            bpm['is_active'] = False
            logger.info(f"Batch processing for stage '{bpm['current_stage']}' stopped due to user cancellation.")
            st.rerun()
        elif not bpm['is_active'] and not bpm['user_cancelled']: # Already inactive, no rerun needed unless state changed
            logger.info(f"Batch processing for stage '{bpm['current_stage']}' is not active. Nothing to do.")
        return

    # Handle Completion (Top)
    if bpm['current_index'] >= bpm['total_files']:
        bpm['is_active'] = False
        logger.info(f"Batch processing for stage '{bpm['current_stage']}' completed. Processed {bpm['successful_count']} successfully, {bpm['error_count']} errors.")
        st.rerun()
        return

    # Sub-batch Loop
    bpm['processed_in_sub_batch'] = 0 # Reset for this iteration

    for i in range(bpm['sub_batch_size']):
        if bpm['current_index'] >= bpm['total_files'] or bpm['user_cancelled']:
            break # Exit sub-batch loop if all files processed or cancelled

        file_info = bpm['files_to_process'][bpm['current_index']]
        file_id = file_info.get('id', f"unknown_id_{bpm['current_index']}")
        file_name = file_info.get('name', f"Unknown File {bpm['current_index'] + 1}")

        logger.info(f"Processing file {bpm['current_index'] + 1}/{bpm['total_files']}: {file_name} (ID: {file_id}) for stage '{bpm['current_stage']}'.")

        try:
            result = stage_processor_fn(file_info, bpm['stage_config'])
            status = result.get('status', 'success') # Default to success if not specified
            message = result.get('message', 'Processed successfully.')
            data = result.get('data')
            record_batch_file_result(file_id, file_name, status, message, data)
        except Exception as e:
            error_message = f"Error processing file {file_name} (ID: {file_id}): {str(e)}"
            logger.error(error_message, exc_info=True)
            record_batch_file_result(file_id, file_name, 'error', error_message, {'exception_type': type(e).__name__})

        bpm['current_index'] += 1
        bpm['processed_in_sub_batch'] += 1

        # API call delay, but not after the very last item of the entire batch or if cancelled
        if bpm['api_call_delay_seconds'] > 0 and bpm['current_index'] < bpm['total_files'] and not bpm['user_cancelled']:
            # Also, don't delay if this was the last item of the current sub-batch iteration
            if i < bpm['sub_batch_size'] - 1 :
                 time.sleep(bpm['api_call_delay_seconds'])


    # After Sub-batch Loop
    if bpm['current_index'] < bpm['total_files'] and not bpm['user_cancelled']:
        if bpm['sub_batch_delay_seconds'] > 0:
            logger.debug(f"Sub-batch completed for stage '{bpm['current_stage']}'. Delaying for {bpm['sub_batch_delay_seconds']}s.")
            time.sleep(bpm['sub_batch_delay_seconds'])
        st.rerun() # Continue processing next sub-batch
    else:
        # Batch truly finished or was cancelled and this iteration processed its last allowed items
        bpm['is_active'] = False
        if bpm['user_cancelled']:
            logger.info(f"Batch processing for stage '{bpm['current_stage']}' concluded after user cancellation. Processed {bpm['successful_count']} successfully, {bpm['error_count']} errors before full stop.")
        else:
            logger.info(f"Batch processing for stage '{bpm['current_stage']}' fully completed. Processed {bpm['successful_count']} successfully, {bpm['error_count']} errors.")
        st.rerun() # Rerun to reflect final inactive state
