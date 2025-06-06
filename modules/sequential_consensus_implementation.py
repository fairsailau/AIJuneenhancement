import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def categorize_document_with_sequential_consensus(
    file_id: str, 
    model1: str, 
    model2: str, 
    model3: str, 
    document_types_with_desc: List[Dict[str, str]],
    disagreement_threshold: float = 0.2
) -> Dict[str, Any]:
    """
    Perform document categorization using sequential consensus approach:
    1. Model 1 performs initial categorization
    2. Model 2 reviews Model 1's results
    3. Model 3 arbitrates if there's significant disagreement
    
    Args:
        file_id: Box file ID
        model1: AI model for initial categorization
        model2: AI model for reviewing Model 1's results
        model3: AI model for arbitration (used only when needed)
        document_types_with_desc: List of document types with descriptions
        disagreement_threshold: Threshold to trigger Model 3 arbitration
        
    Returns:
        Dictionary with categorization results and consensus information
    """
    # Import directly from document_categorization_utils to avoid circular imports
    from modules.document_categorization_utils import (
        categorize_document,
        extract_document_features,
        calculate_multi_factor_confidence,
        apply_confidence_calibration
    )
    
    logger.info(f"Starting sequential consensus categorization for file {file_id}")
    
    # Step 1: Initial categorization with Model 1
    logger.info(f"Step 1: Initial categorization with {model1}")
    model1_result = categorize_document(file_id, model1, document_types_with_desc)
    model1_result["model_name"] = model1
    
    # Extract document features and calculate multi-factor confidence for Model 1
    document_features = extract_document_features(file_id)
    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    
    model1_multi_factor_confidence = calculate_multi_factor_confidence(
        model1_result["confidence"],
        document_features,
        model1_result["document_type"],
        model1_result.get("reasoning", ""),
        valid_categories
    )
    model1_result["multi_factor_confidence"] = model1_multi_factor_confidence
    model1_result["calibrated_confidence"] = apply_confidence_calibration(
        model1_result["document_type"],
        model1_multi_factor_confidence.get("overall", model1_result["confidence"])
    )
    
    # Step 2: First get Model 2's independent assessment, then review
    logger.info(f"Step 2: Independent assessment and review by {model2}")
    model2_result = two_stage_review_categorization(file_id, model2, model1_result, document_types_with_desc)
    model2_result["model_name"] = model2
    
    # Calculate agreement level and confidence
    agreement_level, confidence_adjustment = calculate_agreement_confidence(model1_result, model2_result)
    
    # Determine if arbitration is needed
    needs_arbitration = False
    arbitration_reason = ""
    
    # Check for category disagreement
    if model1_result["document_type"] != model2_result["document_type"]:
        needs_arbitration = True
        arbitration_reason = "Category disagreement"
    
    # Check for significant confidence difference
    elif abs(model1_result["confidence"] - model2_result["confidence"]) > disagreement_threshold:
        needs_arbitration = True
        arbitration_reason = f"Confidence difference exceeds threshold ({disagreement_threshold})"
    
    # Step 3: Arbitration by Model 3 if needed
    model3_result = {}
    final_document_type = ""
    final_confidence = 0.0
    final_reasoning = ""
    
    if needs_arbitration:
        logger.info(f"Step 3: Arbitration needed ({arbitration_reason}). Using {model3}")
        model3_result = arbitrate_categorization(file_id, model3, model1_result, model2_result, document_types_with_desc)
        model3_result["model_name"] = model3
        
        # Use Model 3's decision as final
        final_document_type = model3_result["document_type"]
        final_confidence = model3_result["confidence"]
        final_reasoning = model3_result["reasoning"]
        agreement_level = "Arbitrated"
    else:
        logger.info("No arbitration needed, using Model 2's review as final")
        # Use Model 2's review as final (it already considered Model 1's result)
        final_document_type = model2_result["document_type"]
        final_confidence = model2_result["confidence"]
        final_reasoning = model2_result["reasoning"]
    
    # Calculate final multi-factor confidence
    final_multi_factor_confidence = calculate_multi_factor_confidence(
        final_confidence,
        document_features,
        final_document_type,
        final_reasoning,
        valid_categories
    )
    
    # Apply calibration to final confidence
    final_calibrated_confidence = apply_confidence_calibration(
        final_document_type,
        final_multi_factor_confidence.get("overall", final_confidence)
    )
    
    # Prepare sequential consensus information
    sequential_consensus = {
        "agreement_level": agreement_level,
        "confidence_adjustment": confidence_adjustment,
        "needs_arbitration": needs_arbitration,
        "arbitration_reason": arbitration_reason if needs_arbitration else ""
    }
    
    # Prepare final result
    result = {
        "document_type": final_document_type,
        "confidence": final_confidence,
        "reasoning": final_reasoning,
        "multi_factor_confidence": final_multi_factor_confidence,
        "calibrated_confidence": final_calibrated_confidence,
        "document_features": document_features,
        "sequential_consensus": sequential_consensus,
        "model1_result": model1_result,
        "model2_result": model2_result
    }
    
    # Include Model 3 result if arbitration was used
    if needs_arbitration:
        result["model3_result"] = model3_result
    
    return result

def two_stage_review_categorization(
    file_id: str, 
    model: str, 
    initial_result: Dict[str, Any], 
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Two-stage review process:
    1. First get Model 2's independent assessment
    2. Then have Model 2 review Model 1's results
    
    This approach reduces bias by first getting an independent opinion.
    
    Args:
        file_id: Box file ID
        model: AI model for review
        initial_result: Result from the initial categorization
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with review results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    # STAGE 1: Independent assessment
    independent_prompt = (
        f"You are an expert document reviewer. Please categorize this document into one of the following categories:\n\n"
        f"Available categories:\n{category_options_text}\n\n"
        f"Respond in the following format:\n"
        f"Category: [your selected category name]\n"
        f"Confidence: [your confidence score between 0.0 and 1.0]\n"
        f"Reasoning: [Your detailed reasoning for the categorization]"
    )

    logger.info(f"Box AI Independent Assessment Request for file {file_id} (model: {model}):\n{independent_prompt}")

    api_url = "https://api.box.com/2.0/ai/ask"
    independent_request_body = {
        "mode": "single_item_qa",
        "prompt": independent_prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI independent assessment call for file {file_id} with model {model}")
        independent_response = requests.post(api_url, headers=headers, json=independent_request_body, timeout=180)
        independent_response.raise_for_status()
        independent_data = independent_response.json()
        logger.info(f"Box AI independent assessment response for {file_id}: {json.dumps(independent_data)}")

        # Parse independent assessment
        independent_result = {}
        if "answer" in independent_data and independent_data["answer"]:
            independent_answer = independent_data["answer"]
            independent_result = parse_independent_response(independent_answer, valid_categories)
        else:
            logger.warning(f"No answer field or empty answer in Box AI independent assessment for file {file_id}")
            independent_result = {
                "document_type": "Unknown",
                "confidence": 0.5,
                "reasoning": "Independent assessment failed to provide a response."
            }
        
        # STAGE 2: Review with knowledge of Model 1's results
        initial_category = initial_result["document_type"]
        initial_confidence = initial_result["confidence"]
        initial_reasoning = initial_result.get("reasoning", "No reasoning provided")
        
        independent_category = independent_result["document_type"]
        independent_confidence = independent_result["confidence"]
        independent_reasoning = independent_result["reasoning"]
        
        review_prompt = (
            f"You are an expert document reviewer. You have been asked to review a document categorization.\n\n"
            f"First, you independently categorized this document as:\n"
            f"Category: {independent_category}\n"
            f"Confidence: {independent_confidence:.2f}\n"
            f"Your reasoning: {independent_reasoning}\n\n"
            f"Another AI model categorized the same document as:\n"
            f"Category: {initial_category}\n"
            f"Confidence: {initial_confidence:.2f}\n"
            f"Their reasoning: {initial_reasoning}\n\n"
            f"Please compare these two assessments and provide your final categorization.\n\n"
            f"Available categories:\n{category_options_text}\n\n"
            f"Respond in the following format:\n"
            f"Category: [your final category selection]\n"
            f"Confidence: [your final confidence score between 0.0 and 1.0]\n"
            f"Agreement: [Agree/Partially Agree/Disagree] with the other model's categorization\n"
            f"Assessment: [Your assessment of the other model's categorization and reasoning]\n"
            f"Reasoning: [Your detailed reasoning for your final categorization]"
        )

        logger.info(f"Box AI Review Request for file {file_id} (model: {model}):\n{review_prompt}")

        review_request_body = {
            "mode": "single_item_qa",
            "prompt": review_prompt,
            "items": [{"type": "file", "id": file_id}],
            "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
        }

        logger.info(f"Making Box AI review call for file {file_id} with model {model}")
        review_response = requests.post(api_url, headers=headers, json=review_request_body, timeout=180)
        review_response.raise_for_status()
        review_data = review_response.json()
        logger.info(f"Box AI review response for {file_id}: {json.dumps(review_data)}")

        if "answer" in review_data and review_data["answer"]:
            original_response = review_data["answer"]
            parsed_result = parse_review_response(original_response, valid_categories, initial_result)
            
            # Add independent assessment to result
            parsed_result["independent_assessment"] = independent_result
            
            # Calculate confidence adjustment factors
            confidence_adjustment_factors = {
                "agreement_bonus": 0.0,
                "disagreement_penalty": 0.0,
                "reasoning_quality": 0.0
            }
            
            # Apply adjustments based on agreement level
            if parsed_result["review_assessment"]["agreement_level"] == "Agree":
                confidence_adjustment_factors["agreement_bonus"] = 0.05
            elif parsed_result["review_assessment"]["agreement_level"] == "Partially Agree":
                confidence_adjustment_factors["agreement_bonus"] = 0.02
            else:  # Disagree
                confidence_adjustment_factors["disagreement_penalty"] = -0.05
            
            # Adjust based on reasoning quality assessment
            reasoning_assessment = parsed_result["review_assessment"]["assessment_reasoning"].lower()
            if "excellent" in reasoning_assessment or "thorough" in reasoning_assessment or "comprehensive" in reasoning_assessment:
                confidence_adjustment_factors["reasoning_quality"] = 0.03
            elif "poor" in reasoning_assessment or "inadequate" in reasoning_assessment or "insufficient" in reasoning_assessment:
                confidence_adjustment_factors["reasoning_quality"] = -0.03
            
            parsed_result["confidence_adjustment_factors"] = confidence_adjustment_factors
            
            return parsed_result
        else:
            logger.warning(f"No answer field or empty answer in Box AI review response for file {file_id}. Response: {review_data}")
            return {
                "document_type": independent_result["document_type"],  # Use independent assessment as fallback
                "confidence": independent_result["confidence"] * 0.9,  # Slightly reduce confidence due to review failure
                "reasoning": "Review model did not provide a valid response. Using independent assessment.",
                "original_response": str(review_data),
                "independent_assessment": independent_result,
                "review_assessment": {
                    "agreement_level": "Unknown",
                    "assessment_reasoning": "Review failed"
                }
            }
    except Exception as e:
        logger.error(f"Error during Box AI review call for file {file_id}: {str(e)}")
        return {
            "document_type": initial_result["document_type"],
            "confidence": initial_result["confidence"] * 0.9,  # Slightly reduce confidence due to review failure
            "reasoning": f"Error during review: {str(e)}. Using initial categorization.",
            "original_response": str(e),
            "review_assessment": {
                "agreement_level": "Unknown",
                "assessment_reasoning": f"Review error: {str(e)}"
            }
        }

def parse_independent_response(response_text: str, valid_categories: List[str]) -> Dict[str, Any]:
    """
    Parse the response from the independent assessment.
    
    Args:
        response_text: Response text from the AI model
        valid_categories: List of valid category names
        
    Returns:
        Dictionary with parsed assessment results
    """
    logger.info(f"Parsing independent assessment response: {response_text[:150]}...")
    
    # Default values
    category = "Unknown"
    confidence = 0.5
    reasoning = ""
    
    lines = response_text.strip().split("\n")
    
    # Parse each line based on expected format
    for i, line in enumerate(lines):
        if line.lower().startswith("category:"):
            extracted_category = line.split(":", 1)[1].strip()
            # Find exact or partial match
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    break
            else:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
        
        elif line.lower().startswith("confidence:"):
            confidence_match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
        
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
            # Include all remaining lines as part of reasoning
            if i+1 < len(lines):
                reasoning += "\n" + "\n".join(lines[i+1:])
            break
    
    # If no reasoning found, use everything after the confidence as reasoning
    if not reasoning:
        for i, line in enumerate(lines):
            if line.lower().startswith("confidence:"):
                if i+1 < len(lines):
                    reasoning = "\n".join(lines[i+1:])
                break
    
    # If still no reasoning, use the original response
    if not reasoning:
        reasoning = response_text
    
    logger.info(f"Parsed independent assessment - Category: {category}, Confidence: {confidence:.2f}")
    
    return {
        "document_type": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "original_response": response_text
    }

def parse_review_response(response_text: str, valid_categories: List[str], initial_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse the response from the review model.
    
    Args:
        response_text: Response text from the AI model
        valid_categories: List of valid category names
        initial_result: Result from the initial categorization
        
    Returns:
        Dictionary with parsed review results
    """
    logger.info(f"Parsing review response: {response_text[:150]}...")
    
    # Default to initial result values
    category = initial_result["document_type"]
    confidence = initial_result["confidence"]
    agreement_level = "Unknown"
    assessment_reasoning = ""
    reasoning = ""
    
    lines = response_text.strip().split("\n")
    
    # Parse each line based on expected format
    for i, line in enumerate(lines):
        if line.lower().startswith("category:"):
            extracted_category = line.split(":", 1)[1].strip()
            # Find exact or partial match
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    break
            else:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
        
        elif line.lower().startswith("confidence:"):
            confidence_match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
        
        elif line.lower().startswith("agreement:"):
            agreement_text = line.split(":", 1)[1].strip().lower()
            if "agree" in agreement_text and "partially" not in agreement_text and "dis" not in agreement_text:
                agreement_level = "Agree"
            elif "partially" in agreement_text or "somewhat" in agreement_text:
                agreement_level = "Partially Agree"
            elif "disagree" in agreement_text or "dis agree" in agreement_text:
                agreement_level = "Disagree"
            else:
                agreement_level = "Unknown"
        
        elif line.lower().startswith("assessment:"):
            assessment_reasoning = line.split(":", 1)[1].strip()
            # Check if assessment continues on next lines (until we hit Reasoning:)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("reasoning:"):
                    break
                assessment_reasoning += " " + lines[j]
        
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
            # Include all remaining lines as part of reasoning
            if i+1 < len(lines):
                reasoning += "\n" + "\n".join(lines[i+1:])
            break
    
    # If no reasoning found, use everything after the Assessment as reasoning
    if not reasoning:
        for i, line in enumerate(lines):
            if line.lower().startswith("assessment:"):
                if i+1 < len(lines):
                    reasoning = "\n".join(lines[i+1:])
                break
    
    # If still no reasoning, use the original response
    if not reasoning:
        reasoning = response_text
    
    # If no assessment reasoning found, create a default one
    if not assessment_reasoning:
        if agreement_level == "Agree":
            assessment_reasoning = "The original categorization appears correct."
        elif agreement_level == "Partially Agree":
            assessment_reasoning = "The original categorization is partially correct but could be improved."
        elif agreement_level == "Disagree":
            assessment_reasoning = "The original categorization appears incorrect."
        else:
            assessment_reasoning = "Unable to determine agreement with original categorization."
    
    logger.info(f"Parsed review - Category: {category}, Confidence: {confidence:.2f}, Agreement: {agreement_level}")
    
    return {
        "document_type": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "original_response": response_text,
        "review_assessment": {
            "agreement_level": agreement_level,
            "assessment_reasoning": assessment_reasoning
        }
    }

def arbitrate_categorization(
    file_id: str, 
    model: str, 
    model1_result: Dict[str, Any], 
    model2_result: Dict[str, Any],
    document_types_with_desc: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Have a third model arbitrate between conflicting categorization results.
    
    Args:
        file_id: Box file ID
        model: AI model for arbitration
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        document_types_with_desc: List of document types with descriptions
        
    Returns:
        Dictionary with arbitration results
    """
    access_token = None
    if hasattr(st.session_state.client, "_oauth"):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, "auth") and hasattr(st.session_state.client.auth, "access_token"):
        access_token = st.session_state.client.auth.access_token
    if not access_token:
        raise ValueError("Could not retrieve access token from client")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }

    valid_categories = [dtype["name"] for dtype in document_types_with_desc]
    category_options_text = "\n".join([f"- {dtype['name']}: {dtype['description']}" for dtype in document_types_with_desc])
    
    model1_category = model1_result["document_type"]
    model1_confidence = model1_result["confidence"]
    model1_reasoning = model1_result.get("reasoning", "No reasoning provided")
    
    model2_category = model2_result["document_type"]
    model2_confidence = model2_result["confidence"]
    model2_reasoning = model2_result.get("reasoning", "No reasoning provided")
    
    # Include independent assessment if available
    independent_assessment_text = ""
    if "independent_assessment" in model2_result:
        independent_category = model2_result["independent_assessment"]["document_type"]
        independent_confidence = model2_result["independent_assessment"]["confidence"]
        independent_reasoning = model2_result["independent_assessment"]["reasoning"]
        
        independent_assessment_text = (
            f"MODEL 2 INDEPENDENT ASSESSMENT (before seeing Model 1's results):\n"
            f"Category: {independent_category}\n"
            f"Confidence: {independent_confidence:.2f}\n"
            f"Reasoning: {independent_reasoning}\n\n"
        )
    
    prompt = (
        f"You are an expert document arbitrator. Two AI models have categorized the same document with different results:\n\n"
        f"MODEL 1 CATEGORIZATION:\n"
        f"Category: {model1_category}\n"
        f"Confidence: {model1_confidence:.2f}\n"
        f"Reasoning: {model1_reasoning}\n\n"
        f"MODEL 2 CATEGORIZATION:\n"
        f"{independent_assessment_text}"
        f"Category: {model2_category}\n"
        f"Confidence: {model2_confidence:.2f}\n"
        f"Reasoning: {model2_reasoning}\n\n"
        f"Please examine the document yourself and arbitrate between these conflicting categorizations. You should:\n"
        f"1. Determine which model's categorization is more accurate (or provide your own if both are incorrect)\n"
        f"2. Provide your own confidence score\n"
        f"3. Assess the quality of each model's reasoning\n"
        f"4. Provide your own detailed reasoning\n\n"
        f"Available categories:\n{category_options_text}\n\n"
        f"Respond in the following format:\n"
        f"Category: [your selected category name]\n"
        f"Confidence: [your confidence score between 0.0 and 1.0]\n"
        f"Model 1 Assessment: [Your assessment of Model 1's categorization]\n"
        f"Model 2 Assessment: [Your assessment of Model 2's categorization]\n"
        f"Arbitration: [Your explanation of which model is more accurate and why]\n"
        f"Reasoning: [Your detailed reasoning for the final categorization]"
    )

    logger.info(f"Box AI Arbitration Request for file {file_id} (model: {model}):\n{prompt}")

    api_url = "https://api.box.com/2.0/ai/ask"
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [{"type": "file", "id": file_id}],
        "ai_agent": {"type": "ai_agent_ask", "basic_text": {"model": model, "mode": "default"}}
    }

    try:
        logger.info(f"Making Box AI arbitration call for file {file_id} with model {model}")
        response = requests.post(api_url, headers=headers, json=request_body, timeout=180)
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"Box AI arbitration response for {file_id}: {json.dumps(response_data)}")

        if "answer" in response_data and response_data["answer"]:
            original_response = response_data["answer"]
            parsed_result = parse_arbitration_response(original_response, valid_categories, model1_result, model2_result)
            
            # Calculate confidence factors based on arbitration
            confidence_factors = {
                "model_agreement": 0.0,
                "reasoning_quality": 0.0,
                "arbitration_confidence": 0.0
            }
            
            # Check if arbitration agrees with either model
            if parsed_result["document_type"] == model1_result["document_type"]:
                confidence_factors["model_agreement"] = 0.05
            elif parsed_result["document_type"] == model2_result["document_type"]:
                confidence_factors["model_agreement"] = 0.05
            
            # Assess reasoning quality based on arbitration assessment
            arbitration_reasoning = parsed_result["arbitration_assessment"]["arbitration_reasoning"].lower()
            if "clear" in arbitration_reasoning or "strong" in arbitration_reasoning or "definitive" in arbitration_reasoning:
                confidence_factors["reasoning_quality"] = 0.05
            elif "uncertain" in arbitration_reasoning or "ambiguous" in arbitration_reasoning:
                confidence_factors["reasoning_quality"] = -0.05
            
            # Arbitration confidence factor
            confidence_factors["arbitration_confidence"] = parsed_result["confidence"] * 0.1
            
            parsed_result["confidence_factors"] = confidence_factors
            
            return parsed_result
        else:
            logger.warning(f"No answer field or empty answer in Box AI arbitration response for file {file_id}. Response: {response_data}")
            # Fall back to the model with higher confidence
            if model1_result["confidence"] >= model2_result["confidence"]:
                fallback_result = model1_result
                fallback_message = "Arbitration failed. Using Model 1's result (higher confidence)."
            else:
                fallback_result = model2_result
                fallback_message = "Arbitration failed. Using Model 2's result (higher confidence)."
                
            return {
                "document_type": fallback_result["document_type"],
                "confidence": fallback_result["confidence"] * 0.9,  # Slightly reduce confidence due to arbitration failure
                "reasoning": fallback_message + "\n\nOriginal reasoning: " + fallback_result.get("reasoning", ""),
                "original_response": str(response_data),
                "arbitration_assessment": {
                    "model1_assessment": "Arbitration failed",
                    "model2_assessment": "Arbitration failed",
                    "arbitration_reasoning": fallback_message
                }
            }
    except Exception as e:
        logger.error(f"Error during Box AI arbitration call for file {file_id}: {str(e)}")
        # Fall back to the model with higher confidence
        if model1_result["confidence"] >= model2_result["confidence"]:
            fallback_result = model1_result
            fallback_message = f"Arbitration error: {str(e)}. Using Model 1's result (higher confidence)."
        else:
            fallback_result = model2_result
            fallback_message = f"Arbitration error: {str(e)}. Using Model 2's result (higher confidence)."
            
        return {
            "document_type": fallback_result["document_type"],
            "confidence": fallback_result["confidence"] * 0.9,  # Slightly reduce confidence due to arbitration failure
            "reasoning": fallback_message + "\n\nOriginal reasoning: " + fallback_result.get("reasoning", ""),
            "original_response": str(e),
            "arbitration_assessment": {
                "model1_assessment": "Arbitration error",
                "model2_assessment": "Arbitration error",
                "arbitration_reasoning": fallback_message
            }
        }

def parse_arbitration_response(
    response_text: str, 
    valid_categories: List[str], 
    model1_result: Dict[str, Any], 
    model2_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Parse the response from the arbitration model.
    
    Args:
        response_text: Response text from the AI model
        valid_categories: List of valid category names
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        
    Returns:
        Dictionary with parsed arbitration results
    """
    logger.info(f"Parsing arbitration response: {response_text[:150]}...")
    
    # Default to model with higher confidence
    if model1_result["confidence"] >= model2_result["confidence"]:
        category = model1_result["document_type"]
        confidence = model1_result["confidence"]
    else:
        category = model2_result["document_type"]
        confidence = model2_result["confidence"]
    
    model1_assessment = ""
    model2_assessment = ""
    arbitration_reasoning = ""
    reasoning = ""
    
    lines = response_text.strip().split("\n")
    
    # Parse each line based on expected format
    for i, line in enumerate(lines):
        if line.lower().startswith("category:"):
            extracted_category = line.split(":", 1)[1].strip()
            # Find exact or partial match
            for valid_cat in valid_categories:
                if valid_cat.lower() == extracted_category.lower():
                    category = valid_cat
                    break
            else:
                for valid_cat in valid_categories:
                    if valid_cat.lower() in extracted_category.lower() or extracted_category.lower() in valid_cat.lower():
                        category = valid_cat
                        break
        
        elif line.lower().startswith("confidence:"):
            confidence_match = re.search(r"([0-9]*\.?[0-9]+)", line)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    logger.warning(f"Could not parse confidence value: {line}")
        
        elif line.lower().startswith("model 1 assessment:"):
            model1_assessment = line.split(":", 1)[1].strip()
            # Check if assessment continues on next lines (until we hit another section)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("model 2 assessment:") or \
                   lines[j].lower().startswith("arbitration:") or \
                   lines[j].lower().startswith("reasoning:"):
                    break
                model1_assessment += " " + lines[j]
        
        elif line.lower().startswith("model 2 assessment:"):
            model2_assessment = line.split(":", 1)[1].strip()
            # Check if assessment continues on next lines (until we hit another section)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("arbitration:") or \
                   lines[j].lower().startswith("reasoning:"):
                    break
                model2_assessment += " " + lines[j]
        
        elif line.lower().startswith("arbitration:"):
            arbitration_reasoning = line.split(":", 1)[1].strip()
            # Check if arbitration continues on next lines (until we hit Reasoning:)
            for j in range(i+1, len(lines)):
                if lines[j].lower().startswith("reasoning:"):
                    break
                arbitration_reasoning += " " + lines[j]
        
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()
            # Include all remaining lines as part of reasoning
            if i+1 < len(lines):
                reasoning += "\n" + "\n".join(lines[i+1:])
            break
    
    # If no reasoning found, use everything after the Arbitration as reasoning
    if not reasoning:
        for i, line in enumerate(lines):
            if line.lower().startswith("arbitration:"):
                if i+1 < len(lines):
                    reasoning = "\n".join(lines[i+1:])
                break
    
    # If still no reasoning, use the original response
    if not reasoning:
        reasoning = response_text
    
    # If no model assessments or arbitration reasoning found, create default ones
    if not model1_assessment:
        model1_assessment = f"Model 1 categorized as {model1_result['document_type']} with confidence {model1_result['confidence']:.2f}."
    
    if not model2_assessment:
        model2_assessment = f"Model 2 categorized as {model2_result['document_type']} with confidence {model2_result['confidence']:.2f}."
    
    if not arbitration_reasoning:
        if category == model1_result["document_type"]:
            arbitration_reasoning = "After review, Model 1's categorization appears more accurate."
        elif category == model2_result["document_type"]:
            arbitration_reasoning = "After review, Model 2's categorization appears more accurate."
        else:
            arbitration_reasoning = "After review, neither model's categorization appears fully accurate. A different category has been selected."
    
    logger.info(f"Parsed arbitration - Category: {category}, Confidence: {confidence:.2f}")
    
    return {
        "document_type": category,
        "confidence": confidence,
        "reasoning": reasoning,
        "original_response": response_text,
        "arbitration_assessment": {
            "model1_assessment": model1_assessment,
            "model2_assessment": model2_assessment,
            "arbitration_reasoning": arbitration_reasoning
        }
    }

def calculate_agreement_confidence(model1_result: Dict[str, Any], model2_result: Dict[str, Any]) -> Tuple[str, float]:
    """
    Calculate agreement level and confidence adjustment based on model results.
    
    Args:
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        
    Returns:
        Tuple of (agreement_level, confidence_adjustment)
    """
    # Check if categories match
    categories_match = model1_result["document_type"] == model2_result["document_type"]
    
    # Check confidence difference
    confidence_diff = abs(model1_result["confidence"] - model2_result["confidence"])
    
    # Check review assessment
    review_agreement = model2_result.get("review_assessment", {}).get("agreement_level", "Unknown")
    
    # Determine agreement level
    if categories_match and review_agreement == "Agree" and confidence_diff < 0.1:
        agreement_level = "Full Agreement"
        confidence_adjustment = 0.05  # Boost confidence for full agreement
    elif categories_match and (review_agreement == "Partially Agree" or confidence_diff < 0.2):
        agreement_level = "Partial Agreement"
        confidence_adjustment = 0.02  # Small boost for partial agreement
    else:
        agreement_level = "Disagreement"
        confidence_adjustment = -0.05  # Penalty for disagreement
    
    return agreement_level, confidence_adjustment

def calculate_arbitration_confidence(
    model1_result: Dict[str, Any], 
    model2_result: Dict[str, Any],
    model3_result: Dict[str, Any]
) -> float:
    """
    Calculate confidence adjustment based on arbitration results.
    
    Args:
        model1_result: Result from the initial categorization
        model2_result: Result from the review
        model3_result: Result from the arbitration
        
    Returns:
        Confidence adjustment factor
    """
    # Check if arbitration agrees with either model
    agrees_with_model1 = model3_result["document_type"] == model1_result["document_type"]
    agrees_with_model2 = model3_result["document_type"] == model2_result["document_type"]
    
    # Base confidence is arbitration model's confidence
    base_confidence = model3_result["confidence"]
    
    # Apply adjustments
    if agrees_with_model1 and agrees_with_model2:
        # All models agree - high confidence
        adjustment = 0.05
    elif agrees_with_model1 or agrees_with_model2:
        # Arbitration agrees with one model - moderate confidence
        adjustment = 0.02
    else:
        # Arbitration disagrees with both models - lower confidence
        adjustment = -0.03
    
    return adjustment
