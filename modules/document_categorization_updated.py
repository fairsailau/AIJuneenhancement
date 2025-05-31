import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Import the sequential consensus implementation
from modules.sequential_consensus_implementation import categorize_document_with_sequential_consensus
from modules.document_categorization_utils import (
    categorize_document,
    extract_document_features,
    calculate_multi_factor_confidence,
    apply_confidence_calibration,
    combine_categorization_results,
    categorize_document_detailed
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

UPDATED_MODEL_LIST = [
    "azure__openai__gpt_4_1_mini", "azure__openai__gpt_4_1", "azure__openai__gpt_4o_mini", "azure__openai__gpt_4o",
    "google__gemini_2_5_pro_preview", "google__gemini_2_5_flash_preview", "google__gemini_2_0_flash_001", "google__gemini_2_0_flash_lite_preview",
    "aws__claude_3_haiku", "aws__claude_3_sonnet", "aws__claude_3_5_sonnet", "aws__claude_3_7_sonnet",
    "aws__claude_4_sonnet", "aws__claude_4_opus", "aws__titan_text_lite",
    "ibm__llama_3_2_90b_vision_instruct", "ibm__llama_4_scout",
    "xai__grok_3_beta", "xai__grok_3_mini_reasoning_beta", "azure__openai__gpt_o3"
]

def document_categorization():
    """
    Main function for document categorization tab.
    """
    st.title("Document Categorization")
    
    # Initialize session state for document categorization
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": [],
            "errors": []
        }
    
    # Initialize document types if not already in session state
    if "document_types" not in st.session_state:
        logger.warning("Initializing/Resetting document_types in session state to default structure.")
        st.session_state.document_types = [
            {"name": "Sales Contract", "description": "Contracts related to sales agreements and terms."},
            {"name": "Invoice", "description": "Billing documents issued by a seller to a buyer, indicating quantities, prices for products or services."},
            {"name": "Tax", "description": "Documents related to government taxation (e.g., tax forms, filings, receipts)."},
            {"name": "Financial Report", "description": "Reports detailing the financial status or performance of an entity."},
            {"name": "Employment Contract", "description": "Agreements outlining terms and conditions of employment."},
            {"name": "PII", "description": "Documents containing Personally Identifiable Information that needs careful handling."},
            {"name": "Other", "description": "Any document not fitting into the specific categories above."}
        ]
    
    # Create tabs for categorization and settings
    tab1, tab2 = st.tabs(["Categorization", "Settings"])
    
    with tab1: # Categorization Tab (Indent level 1: 4 spaces)
        st.write("## AI Model Selection") # Indent level 2: 8 spaces
        
        # Consensus mode selection
        st.write("### Consensus Mode") # Indent level 2: 8 spaces
        consensus_mode = st.radio( # Indent level 2: 8 spaces
            "Consensus Mode",
            ["Standard", "Parallel Consensus", "Sequential Consensus"],
            help="Standard: Single model categorization. Parallel: Multiple models categorize independently. Sequential: Models review each other's work."
        )
        
        if consensus_mode == "Standard": # Indent level 2: 8 spaces
            # Single model selection
            default_standard_model = "google__gemini_2_0_flash_001" # Indent level 3: 12 spaces
            if default_standard_model not in UPDATED_MODEL_LIST: # Indent level 3: 12 spaces
                default_standard_model = UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None # Indent level 4: 16 spaces

            model = st.selectbox( # Indent level 3: 12 spaces
                "Select AI Model",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_standard_model) if default_standard_model in UPDATED_MODEL_LIST else 0,
                help="Select the AI model to use for document categorization."
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox( # Indent level 3: 12 spaces
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage: # Indent level 3: 12 spaces
                confidence_threshold = st.slider( # Indent level 4: 16 spaces
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else: # Indent level 3: 12 spaces
                confidence_threshold = 0.6  # Default value # Indent level 4: 16 spaces
        
        elif consensus_mode == "Parallel Consensus": # Indent level 2: 8 spaces
            st.write("Select models for parallel consensus:") # Indent level 3: 12 spaces
            
            default_parallel_models = [ # Indent level 3: 12 spaces
                m for m in ["google__gemini_2_0_flash_001", "aws__claude_3_sonnet", "azure__openai__gpt_4_1"]
                if m in UPDATED_MODEL_LIST
            ]
            if not default_parallel_models and UPDATED_MODEL_LIST: # Indent level 3: 12 spaces
                default_parallel_models = [UPDATED_MODEL_LIST[0]] # Indent level 4: 16 spaces

            models = st.multiselect( # Indent level 3: 12 spaces
                "Select models for parallel consensus:",
                options=UPDATED_MODEL_LIST,
                default=default_parallel_models
            )
            
            use_two_stage = st.checkbox( # Indent level 3: 12 spaces
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage: # Indent level 3: 12 spaces
                confidence_threshold = st.slider( # Indent level 4: 16 spaces
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else: # Indent level 3: 12 spaces
                confidence_threshold = 0.6  # Default value # Indent level 4: 16 spaces
        
        else:  # Sequential Consensus # Indent level 2: 8 spaces
            st.write("### Select models for sequential consensus:") # Indent level 3: 12 spaces
            
            # Model 1 (Initial Analysis)
            st.write("#### Model 1 (Initial Analysis)") # Indent level 3: 12 spaces
            default_model1 = "google__gemini_2_0_flash_001" # Indent level 3: 12 spaces
            if default_model1 not in UPDATED_MODEL_LIST: # Indent level 3: 12 spaces
                default_model1 = UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None # Indent level 4: 16 spaces
            model1 = st.selectbox( # Indent level 3: 12 spaces
                "Model 1 (Initial Analysis)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model1) if default_model1 in UPDATED_MODEL_LIST else 0,
                help="This model will perform the initial document categorization."
            )
            
            # Model 2 (Expert Review)
            st.write("#### Model 2 (Expert Review)") # Indent level 3: 12 spaces
            default_model2 = "aws__claude_3_sonnet" # Indent level 3: 12 spaces
            if default_model2 not in UPDATED_MODEL_LIST: # Indent level 3: 12 spaces
                default_model2 = UPDATED_MODEL_LIST[1] if len(UPDATED_MODEL_LIST) > 1 else (UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None) # Indent level 4: 16 spaces
            model2 = st.selectbox( # Indent level 3: 12 spaces
                "Model 2 (Expert Review)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model2) if default_model2 in UPDATED_MODEL_LIST else 0,
                help="This model will review Model 1's categorization."
            )
            
            # Model 3 will be used for arbitration only when needed
            st.write("#### Model 3 will be used for arbitration only when needed:") # Indent level 3: 12 spaces
            
            # Model 3 (Arbitration)
            default_model3 = "aws__claude_3_5_sonnet" # Indent level 3: 12 spaces
            if default_model3 not in UPDATED_MODEL_LIST: # Indent level 3: 12 spaces
                default_model3 = UPDATED_MODEL_LIST[2] if len(UPDATED_MODEL_LIST) > 2 else (UPDATED_MODEL_LIST[0] if UPDATED_MODEL_LIST else None) # Indent level 4: 16 spaces
            model3 = st.selectbox( # Indent level 3: 12 spaces
                "Model 3 (Arbitration)",
                UPDATED_MODEL_LIST,
                index=UPDATED_MODEL_LIST.index(default_model3) if default_model3 in UPDATED_MODEL_LIST else 0,
                help="This model will arbitrate if there's significant disagreement between Models 1 and 2."
            )
            
            # Sequential Consensus Parameters
            st.write("### Sequential Consensus Parameters") # Indent level 3: 12 spaces
            
            # Disagreement threshold
            disagreement_threshold = st.slider( # Indent level 3: 12 spaces
                "Disagreement Threshold",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.01,
                help="Confidence difference threshold that triggers Model 3 arbitration."
            )
            
            # Two-stage categorization option
            use_two_stage = st.checkbox( # Indent level 3: 12 spaces
                "Use two-stage categorization",
                help="First categorize as PII/non-PII, then apply specific categories to non-PII documents."
            )
            
            if use_two_stage: # Indent level 3: 12 spaces
                confidence_threshold = st.slider( # Indent level 4: 16 spaces
                    "Confidence threshold for second-stage",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.01,
                    help="Documents with first-stage confidence below this threshold will undergo second-stage categorization."
                )
            else: # Indent level 3: 12 spaces
                confidence_threshold = 0.6  # Default value # Indent level 4: 16 spaces
        
        st.write("## Categorization Options") # Indent level 2: 8 spaces
        
        source_option = st.radio( # Indent level 2: 8 spaces
            "Select document source for categorization:",
            ("Selected in File Browser", "Specific Folder ID"),
            key="categorization_source_option"
        )

        folder_id_input = "323454589704" # Default value # Indent level 2: 8 spaces
        if source_option == "Specific Folder ID": # Indent level 2: 8 spaces
            folder_id_input = st.text_input("Box Folder ID", value=folder_id_input) # Indent level 3: 12 spaces
        
        # Start and cancel buttons
        col1, col2 = st.columns(2) # Indent level 2: 8 spaces
        with col1: # Indent level 2: 8 spaces
            start_button = st.button("Start Categorization") # Indent level 3: 12 spaces
        with col2: # Indent level 2: 8 spaces
            cancel_button = st.button("Cancel Categorization") # Indent level 3: 12 spaces
        
        # Process categorization
        if start_button: # Indent level 2: 8 spaces
            st.session_state.document_categorization["is_categorized"] = False # Indent level 3: 12 spaces
            st.session_state.document_categorization["results"] = [] # Indent level 3: 12 spaces
            st.session_state.document_categorization["errors"] = [] # Indent level 3: 12 spaces
            
            files_to_process = [] # Indent level 3: 12 spaces
            proceed_with_categorization = False # Indent level 3: 12 spaces

            if source_option == "Selected in File Browser": # Indent level 3: 12 spaces
                selected_file_ids = st.session_state.get("selected_files", []) # Indent level 4: 16 spaces
                selected_folder_ids = st.session_state.get("selected_folders", []) # Indent level 4: 16 spaces
                
                if not selected_file_ids and not selected_folder_ids: # Indent level 4: 16 spaces
                    st.warning("No files or folders selected in File Browser. Please select items in Step 2: Select Files (if available in the app) or choose 'Specific Folder ID'.") # Indent level 5: 20 spaces
                else: # Indent level 4: 16 spaces
                    temp_file_objects = {} # Indent level 5: 20 spaces
                    try: # Indent level 5: 20 spaces
                        if selected_file_ids: # Indent level 6: 24 spaces
                            st.write(f"Processing {len(selected_file_ids)} individually selected files...") # Indent level 7: 28 spaces
                            for file_id in selected_file_ids: # Indent level 7: 28 spaces
                                if file_id not in temp_file_objects: # Indent level 8: 32 spaces
                                    file_obj = st.session_state.client.file(file_id).get() # Indent level 9: 36 spaces
                                    temp_file_objects[file_id] = file_obj # Indent level 9: 36 spaces

                        if selected_folder_ids: # Indent level 6: 24 spaces
                            st.write(f"Processing files from {len(selected_folder_ids)} selected folders...") # Indent level 7: 28 spaces
                            for folder_id_val in selected_folder_ids: # Indent level 7: 28 spaces
                                folder_obj = st.session_state.client.folder(folder_id_val).get() # Indent level 8: 32 spaces
                                st.write(f"Fetching items from folder: {folder_obj.name}") # Indent level 8: 32 spaces
                                items_in_folder = folder_obj.get_items() # Indent level 8: 32 spaces
                                for item in items_in_folder: # Indent level 8: 32 spaces
                                    if item.type == "file" and item.id not in temp_file_objects: # Indent level 9: 36 spaces
                                        temp_file_objects[item.id] = item # Indent level 10: 40 spaces

                        files_to_process = list(temp_file_objects.values()) # Indent level 6: 24 spaces
                        if files_to_process: # Indent level 6: 24 spaces
                            st.info(f"Found {len(files_to_process)} unique files from File Browser selections.") # Indent level 7: 28 spaces
                            proceed_with_categorization = True # Indent level 7: 28 spaces
                        else: # Indent level 6: 24 spaces
                            st.warning("No processable files found in the selected items from File Browser.") # Indent level 7: 28 spaces

                    except Exception as e: # Indent level 5: 20 spaces
                        st.error(f"Error accessing selected items: {str(e)}") # Indent level 6: 24 spaces
                        logger.error(f"Error processing selections from File Browser: {str(e)}") # Indent level 6: 24 spaces

            elif source_option == "Specific Folder ID": # Indent level 3: 12 spaces
                if not folder_id_input: # Indent level 4: 16 spaces
                    st.warning("Please enter a Box Folder ID.") # Indent level 5: 20 spaces
                else: # Indent level 4: 16 spaces
                    try: # Indent level 5: 20 spaces
                        folder = st.session_state.client.folder(folder_id_input).get() # Indent level 6: 24 spaces
                        items = folder.get_items() # Indent level 6: 24 spaces
                        files_to_process = [item for item in items if item.type == "file"] # Indent level 6: 24 spaces
                        if files_to_process: # Indent level 6: 24 spaces
                             st.info(f"Found {len(files_to_process)} files in folder '{folder.name}'.") # Indent level 7: 28 spaces
                             proceed_with_categorization = True # Indent level 7: 28 spaces
                        else: # Indent level 6: 24 spaces
                            st.warning(f"No files found in the folder ID: {folder_id_input}") # Indent level 7: 28 spaces
                    except Exception as e: # Indent level 5: 20 spaces
                        st.error(f"Error accessing folder ID {folder_id_input}: {str(e)}") # Indent level 6: 24 spaces
                        logger.error(f"Error accessing folder {folder_id_input}: {str(e)}") # Indent level 6: 24 spaces

            if proceed_with_categorization and files_to_process: # Indent level 3: 12 spaces
                progress_text = st.empty() # Indent level 4: 16 spaces

                if consensus_mode == "Standard": # Indent level 4: 16 spaces
                    progress_text.info(f"Processing {len(files_to_process)} files with {model}...") # Indent level 5: 20 spaces

                    for file_obj in files_to_process: # Indent level 5: 20 spaces
                        try: # Indent level 6: 24 spaces
                            progress_text.info(f"Processing {file_obj.name}...") # Indent level 7: 28 spaces
                                
                                if use_two_stage: # Indent level 7: 28 spaces
                                    result = categorize_document_detailed( # Indent level 8: 32 spaces
                                        file_obj.id,
                                        model, 
                                        st.session_state.document_types,
                                        confidence_threshold
                                    )
                                else: # Indent level 7: 28 spaces
                                    result = categorize_document( # Indent level 8: 32 spaces
                                        file_obj.id,
                                        model, 
                                        st.session_state.document_types
                                    )
                                
                                result["file_id"] = file_obj.id # Indent level 7: 28 spaces
                                result["file_name"] = file_obj.name # Indent level 7: 28 spaces
                                
                                st.session_state.document_categorization["results"].append(result) # Indent level 7: 28 spaces
                                
                            except Exception as e: # Indent level 6: 24 spaces
                                logger.error(f"Error categorizing document {file_obj.name}: {str(e)}") # Indent level 7: 28 spaces
                                st.session_state.document_categorization["errors"].append({ # Indent level 7: 28 spaces
                                    "file_id": file_obj.id,
                                    "file_name": file_obj.name,
                                    "error": str(e)
                                })
                    
                    elif consensus_mode == "Parallel Consensus": # Indent level 4: 16 spaces
                        if not models: # Indent level 5: 20 spaces
                            st.error("Please select at least one model for parallel consensus.") # Indent level 6: 24 spaces
                        else: # Indent level 5: 20 spaces
                            progress_text.info(f"Processing {len(files_to_process)} files with {len(models)} models in parallel...") # Indent level 6: 24 spaces

                            for file_obj in files_to_process: # Indent level 6: 24 spaces
                                try: # Indent level 7: 28 spaces
                                    progress_text.info(f"Processing {file_obj.name} with parallel consensus...") # Indent level 8: 32 spaces
                                
                                    model_results = [] # Indent level 8: 32 spaces
                                    for model_name_selected in models: # Indent level 8: 32 spaces
                                        try: # Indent level 9: 36 spaces
                                            if use_two_stage: # Indent level 10: 40 spaces
                                                model_result = categorize_document_detailed( # Indent level 11: 44 spaces
                                                    file_obj.id,
                                                    model_name_selected,
                                                    st.session_state.document_types,
                                                    confidence_threshold
                                                )
                                            else: # Indent level 10: 40 spaces
                                                model_result = categorize_document( # Indent level 11: 44 spaces
                                                    file_obj.id,
                                                    model_name_selected,
                                                    st.session_state.document_types
                                                )

                                            model_result["model_name"] = model_name_selected # Indent level 10: 40 spaces
                                            model_results.append(model_result) # Indent level 10: 40 spaces
                                        except Exception as e: # Indent level 9: 36 spaces
                                            logger.error(f"Error with model {model_name_selected} for {file_obj.name}: {str(e)}") # Indent level 10: 40 spaces
                                    
                                    if model_results: # Indent level 8: 32 spaces
                                        current_valid_categories = [dtype["name"] for dtype in st.session_state.document_types] # Indent level 9: 36 spaces
                                        combined_result = combine_categorization_results( # Indent level 9: 36 spaces
                                            model_results,
                                            current_valid_categories,
                                            models
                                        )

                                        combined_result["file_id"] = file_obj.id # Indent level 9: 36 spaces
                                        combined_result["file_name"] = file_obj.name # Indent level 9: 36 spaces
                                        combined_result["model_results"] = model_results # Indent level 9: 36 spaces

                                        st.session_state.document_categorization["results"].append(combined_result) # Indent level 9: 36 spaces
                                    else: # Indent level 8: 32 spaces
                                        st.session_state.document_categorization["errors"].append({ # Indent level 9: 36 spaces
                                            "file_id": file_obj.id,
                                            "file_name": file_obj.name,
                                            "error": "All selected parallel models failed to categorize this document."
                                        })
                                    
                                except Exception as e: # Indent level 7: 28 spaces
                                    logger.error(f"Error categorizing document {file_obj.name} with parallel consensus: {str(e)}") # Indent level 8: 32 spaces
                                    st.session_state.document_categorization["errors"].append({ # Indent level 8: 32 spaces
                                        "file_id": file_obj.id,
                                        "file_name": file_obj.name,
                                        "error": str(e)
                                    })
                    
                    else:  # Sequential Consensus # Indent level 4: 16 spaces
                        progress_text.info(f"Processing {len(files_to_process)} files with sequential consensus...") # Indent level 5: 20 spaces
                        
                        for file_obj in files_to_process: # Indent level 5: 20 spaces
                            try: # Indent level 6: 24 spaces
                                progress_text.info(f"Processing {file_obj.name} with sequential consensus...") # Indent level 7: 28 spaces
                                
                                result = categorize_document_with_sequential_consensus( # Indent level 7: 28 spaces
                                    file_obj.id,
                                    model1,
                                    model2,
                                    model3,
                                    st.session_state.document_types,
                                    disagreement_threshold
                                )
                                
                                result["file_id"] = file_obj.id # Indent level 7: 28 spaces
                                result["file_name"] = file_obj.name # Indent level 7: 28 spaces
                                
                                st.session_state.document_categorization["results"].append(result) # Indent level 7: 28 spaces
                                
                            except Exception as e: # Indent level 6: 24 spaces
                                logger.error(f"Error categorizing document {file_obj.name} with sequential consensus: {str(e)}") # Indent level 7: 28 spaces
                                st.session_state.document_categorization["errors"].append({ # Indent level 7: 28 spaces
                                    "file_id": file_obj.id,
                                    "file_name": file_obj.name,
                                    "error": str(e)
                                })
                    
                if progress_text: progress_text.empty() # Indent level 4: 16 spaces
                st.session_state.document_categorization["is_categorized"] = True # Indent level 4: 16 spaces
                num_processed = len(st.session_state.document_categorization["results"]) # Indent level 4: 16 spaces
                num_errors = len(st.session_state.document_categorization["errors"]) # Indent level 4: 16 spaces

                if num_processed == 0 and num_errors == 0 and not proceed_with_categorization: # Indent level 4: 16 spaces
                    pass # Indent level 5: 20 spaces
                elif num_errors == 0 and num_processed > 0: # Indent level 4: 16 spaces
                    st.success(f"Categorization complete! Processed {num_processed} files.") # Indent level 5: 20 spaces
                elif num_processed > 0 and num_errors > 0: # Indent level 4: 16 spaces
                    st.warning(f"Categorization complete! Processed {num_processed} files with {num_errors} errors.") # Indent level 5: 20 spaces
                elif num_processed == 0 and num_errors > 0 : # Indent level 4: 16 spaces
                    st.error(f"Categorization failed for all {num_errors} attempted items.") # Indent level 5: 20 spaces

        if st.session_state.document_categorization.get("is_categorized", False): # Indent level 2: 8 spaces
            display_categorization_results() # Indent level 3: 12 spaces
    
    with tab2: # Settings Tab # Indent level 1: 4 spaces
        st.write("### Settings") # Indent level 2: 8 spaces
        st.write("#### Document Types Configuration") # Indent level 2: 8 spaces
        configure_document_types() # Indent level 2: 8 spaces

        st.write("#### Confidence Configuration") # Indent level 2: 8 spaces
        configure_confidence_thresholds() # Indent level 2: 8 spaces
        with st.expander("Confidence Validation", expanded=False): # Indent level 2: 8 spaces
            validate_confidence_with_examples() # Indent level 3: 12 spaces

# --- UI Helper Functions (Settings, Results Display) ---

def configure_document_types():
    """
    UI for configuring document types.
    """
    st.write("Define the categories you want to use for document classification.")
    
    if "document_types" not in st.session_state:
        st.session_state.document_types = []
    
    for i, doc_type in enumerate(st.session_state.document_types):
        col1, col2, col3 = st.columns([3, 5, 1])
        with col1:
            new_name = st.text_input(f"Name {i+1}", value=doc_type["name"], key=f"doc_type_name_{i}")
        with col2:
            new_desc = st.text_input(f"Description {i+1}", value=doc_type["description"], key=f"doc_type_desc_{i}")
        with col3:
            if st.button("❌", key=f"remove_doc_type_{i}", help="Remove this document type"):
                st.session_state.document_types.pop(i)
                st.rerun()
                
        st.session_state.document_types[i]["name"] = new_name
        st.session_state.document_types[i]["description"] = new_desc
    
    st.write("#### Add New Document Type")
    col1, col2, col3 = st.columns([3, 5, 1])
    with col1:
        new_name = st.text_input("Name", key="new_doc_type_name")
    with col2:
        new_desc = st.text_input("Description", key="new_doc_type_desc")
    with col3:
        if st.button("➕", key="add_doc_type", help="Add this document type"):
            if new_name:
                st.session_state.document_types.append({
                    "name": new_name,
                    "description": new_desc
                })
                st.rerun()

def configure_confidence_thresholds():
    """
    UI for configuring confidence thresholds.
    """
    st.write("Configure confidence thresholds for automatic actions.")
    
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    st.session_state.confidence_thresholds["auto_accept"] = st.slider(
        "Auto-accept threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds["auto_accept"],
        step=0.01,
        help="Documents with confidence above this threshold will be automatically accepted."
    )
    
    st.session_state.confidence_thresholds["verification"] = st.slider(
        "Verification threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds["verification"],
        step=0.01,
        help="Documents with confidence above this threshold but below auto-accept will require verification."
    )
    
    st.session_state.confidence_thresholds["rejection"] = st.slider(
        "Rejection threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_thresholds["rejection"],
        step=0.01,
        help="Documents with confidence below this threshold will be automatically rejected."
    )

def validate_confidence_with_examples():
    """
    UI for validating confidence thresholds with examples.
    """
    st.write("Test confidence thresholds with example values.")
    
    test_confidence = st.slider(
        "Test confidence value",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.01
    )
    
    if test_confidence >= st.session_state.confidence_thresholds["auto_accept"]:
        status = "Auto-Accepted"
        color = "green"
    elif test_confidence >= st.session_state.confidence_thresholds["verification"]:
        status = "Needs Verification"
        color = "orange"
    elif test_confidence >= st.session_state.confidence_thresholds["rejection"]:
        status = "Low Confidence"
        color = "red"
    else:
        status = "Auto-Rejected"
        color = "red"
    
    st.markdown(f"Status: <span style='color:{color}'>{status}</span>", unsafe_allow_html=True)

def display_categorization_results():
    """
    Display categorization results in a table.
    """
    st.write("## Categorization Results")
    
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        if st.session_state.document_categorization["results"]:
            data = []
            for result in st.session_state.document_categorization["results"]:
                confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
                
                if confidence >= st.session_state.confidence_thresholds["auto_accept"]:
                    status = "Auto-Accepted"
                elif confidence >= st.session_state.confidence_thresholds["verification"]:
                    status = "Needs Verification"
                elif confidence >= st.session_state.confidence_thresholds["rejection"]:
                    status = "Low Confidence"
                else:
                    status = "Auto-Rejected"
                
                consensus_info = ""
                if "model_results" in result:
                    consensus_info = f"Parallel: {len(result['model_results'])} models"
                elif "sequential_consensus" in result:
                    agreement_level = result["sequential_consensus"].get("agreement_level", "Unknown")
                    consensus_info = f"Sequential: {agreement_level}"
                
                data.append({
                    "File Name": result["file_name"],
                    "Document Type": result["document_type"],
                    "Confidence": round(confidence, 2),
                    "Confidence Level": "High" if confidence >= 0.8 else "Medium" if confidence >= 0.6 else "Low",
                    "Status": status,
                    "Consensus": consensus_info
                })
            
            df = pd.DataFrame(data)
            
            def color_status(val):
                if val == "Auto-Accepted":
                    return 'background-color: #c6efce; color: #006100'
                elif val == "Needs Verification":
                    return 'background-color: #ffeb9c; color: #9c5700'
                else:
                    return 'background-color: #ffc7ce; color: #9c0006'
            
            st.dataframe(
                df.style.map(color_status, subset=["Status"]),
                use_container_width=True
            )
        else:
            st.info("No categorization results available.")
    
    with tab2:
        if st.session_state.document_categorization["results"]:
            file_names = [result["file_name"] for result in st.session_state.document_categorization["results"]]
            selected_file = st.selectbox("Select a file to view details", file_names)
            
            selected_result = next(
                (r for r in st.session_state.document_categorization["results"] if r["file_name"] == selected_file),
                None
            )
            
            if selected_result:
                display_detailed_result(selected_result)
        else:
            st.info("No categorization results available.")

def display_detailed_result(result):
    """
    Display detailed categorization result for a single file.
    """
    st.markdown(f"# {result['file_name']} - {result['document_type']}")
    
    confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
    confidence_color = "green" if confidence >= 0.8 else "orange" if confidence >= 0.6 else "red"
    st.markdown(f"## Overall Confidence: <span style='color:{confidence_color}'>{'High' if confidence >= 0.8 else 'Medium' if confidence >= 0.6 else 'Low'} ({confidence:.2f})</span>", unsafe_allow_html=True)
    
    if "sequential_consensus" in result:
        st.markdown("## Sequential Consensus Details")
        
        agreement_level = result["sequential_consensus"].get("agreement_level", "Unknown")
        agreement_color = "green" if agreement_level == "Full Agreement" else "orange" if agreement_level == "Partial Agreement" else "red"
        st.markdown(f"### Agreement Level: <span style='color:{agreement_color}'>{agreement_level}</span>", unsafe_allow_html=True)
        
        model_tabs = st.tabs(["Model 1 (Initial)", "Model 2 (Review)", "Model 3 (Arbitration)"])
        
        with model_tabs[0]:
            if "model1_result" in result:
                model1 = result["model1_result"]
                st.markdown(f"### Model: {model1.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model1.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model1.get('confidence', 0.0):.2f}")
                st.markdown("### Reasoning:")
                st.markdown(model1.get("reasoning", "No reasoning provided"))
        
        with model_tabs[1]:
            if "model2_result" in result:
                model2_details = result["model2_result"]
                if "independent_assessment" in model2_details and isinstance(model2_details["independent_assessment"], dict):
                    independent_assessment = model2_details["independent_assessment"]
                    st.markdown("#### Model 2: Independent Initial Assessment")
                    st.markdown(f"Independent Category: {independent_assessment.get('document_type', 'N/A')}")
                    st.markdown(f"Independent Confidence: {independent_assessment.get('confidence', 0.0):.2f}")
                    st.markdown("Independent Reasoning:")
                    st.markdown(independent_assessment.get('reasoning', 'No reasoning provided'))
                    st.markdown("---")
                    st.markdown("#### Model 2: Final Review Assessment (after seeing Model 1)")
                else:
                    st.markdown("#### Model 2: Review Assessment")
                st.markdown(f"### Model: {model2_details.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model2_details.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model2_details.get('confidence', 0.0):.2f}")
                st.markdown("### Review Assessment Details:")
                if "review_assessment" in model2_details:
                    st.markdown(f"**Agreement Level:** {model2_details['review_assessment'].get('agreement_level', 'Unknown')}")
                    st.markdown(f"**Assessment Reasoning:** {model2_details['review_assessment'].get('assessment_reasoning', 'No assessment provided')}")
                st.markdown("### Confidence Adjustment Factors:")
                if "confidence_adjustment_factors" in model2_details:
                    factors = model2_details["confidence_adjustment_factors"]
                    st.markdown(f"* Agreement Bonus: {factors.get('agreement_bonus', 0.0):.2f}")
                    st.markdown(f"* Disagreement Penalty: {factors.get('disagreement_penalty', 0.0):.2f}")
                    st.markdown(f"* Reasoning Quality: {factors.get('reasoning_quality', 0.0):.2f}")
                st.markdown("### Final Reasoning:")
                st.markdown(model2_details.get("reasoning", "No reasoning provided"))
        
        with model_tabs[2]:
            if "model3_result" in result:
                model3 = result["model3_result"]
                st.markdown(f"### Model: {model3.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model3.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model3.get('confidence', 0.0):.2f}")
                if "arbitration_assessment" in model3:
                    st.markdown("### Arbitration Assessment:")
                    st.markdown(f"#### Model 1 Assessment: {model3['arbitration_assessment'].get('model1_assessment', 'No assessment provided')}")
                    st.markdown(f"#### Model 2 Assessment: {model3['arbitration_assessment'].get('model2_assessment', 'No assessment provided')}")
                    st.markdown(f"#### Arbitration Reasoning: {model3['arbitration_assessment'].get('arbitration_reasoning', 'No reasoning provided')}")
                st.markdown("### Reasoning:")
                st.markdown(model3.get("reasoning", "No reasoning provided"))
            else:
                st.markdown("### No arbitration was needed")
                st.markdown("Model 3 was not used because there was sufficient agreement between Models 1 and 2.")
    
    elif "model_results" in result:
        st.markdown("## Parallel Consensus Details")
        model_names = [model.get("model_name", f"Model {i+1}") for i, model in enumerate(result["model_results"])]
        model_tabs = st.tabs(model_names)
        for i, (tab, model_result) in enumerate(zip(model_tabs, result["model_results"])):
            with tab:
                st.markdown(f"### Model: {model_result.get('model_name', 'Unknown')}")
                st.markdown(f"### Category: {model_result.get('document_type', 'Unknown')}")
                st.markdown(f"### Confidence: {model_result.get('confidence', 0.0):.2f}")
                st.markdown("### Reasoning:")
                st.markdown(model_result.get("reasoning", "No reasoning provided"))
    
    else:
        st.markdown("### Confidence Breakdown")
        multi_factor_confidence = result.get("multi_factor_confidence", {})
        if multi_factor_confidence:
            factors = [
                {"factor": "AI Model", "value": multi_factor_confidence.get("ai_reported", 0.0)},
                {"factor": "Response Quality", "value": multi_factor_confidence.get("response_quality", 0.0)},
                {"factor": "Category Specificity", "value": multi_factor_confidence.get("category_specificity", 0.0)},
                {"factor": "Reasoning Quality", "value": multi_factor_confidence.get("reasoning_quality", 0.0)},
                {"factor": "Document Features Match", "value": multi_factor_confidence.get("document_features_match", 0.0)}
            ]
            for factor in factors:
                factor_value = factor["value"]
                factor_color = "green" if factor_value >= 0.8 else "orange" if factor_value >= 0.6 else "red"
                st.markdown(f"{factor['factor']}: <span style='color:{factor_color}'>{factor_value:.2f}</span>", unsafe_allow_html=True)
                st.progress(factor_value)
        
        if result.get("first_stage_type"):
            st.markdown("### Two-Stage Analysis")
            st.markdown(f"**First Stage Category:** {result['first_stage_type']}")
            st.markdown(f"**First Stage Confidence:** {result.get('first_stage_confidence', 0.0):.2f}")
            st.markdown("**Second Stage Analysis:** Performed due to low initial confidence")
    
    st.markdown("### Reasoning")
    st.markdown(result.get("reasoning", "No reasoning provided"))
    
    if "document_features" in result:
        st.markdown("### Document Features")
        features = result["document_features"]
        for key, value in features.items():
            st.markdown(f"**{key}:** {value}")
