#!/usr/bin/env python3
"""
Marketing Image Generator - Maytapi WhatsApp Version
--------------------------------------------
Production version with Maytapi WhatsApp API integration
Configured for Railway.com deployment
"""

import os
import base64
import json
import logging
import time
import traceback
import hashlib
import math
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Tuple
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests
from PIL import ImageDraw

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")

# Optional Imgur API variables for public image hosting
IMGUR_CLIENT_ID = os.getenv('IMGUR_CLIENT_ID', '')
USE_IMGUR = IMGUR_CLIENT_ID != ''

# Also add print statements for important logs to ensure they appear in Railway logs
def log_and_print(level, message):
    if level == "INFO":
        logger.info(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "DEBUG":
        logger.debug(message)
    
    # Always print to stdout for Railway logs
    print(f"[{level}] {message}")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)

# Store processed messages to prevent duplicates
processed_messages = {}

# Store last message time for rate limiting (per user)
last_message_time = {}

# Initialize OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    log_and_print("ERROR", "OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

log_and_print("INFO", f"Using OpenAI API key starting with: {OPENAI_API_KEY[:8]}...")

# Initialize OpenAI client
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
log_and_print("INFO", "OpenAI client initialized with API key")
print(f"OpenAI API Key starts with: {OPENAI_API_KEY[:5]}..." if OPENAI_API_KEY else "OpenAI API Key is missing!")

# Maytapi configuration - from environment variables
MAYTAPI_API_TOKEN = os.getenv('MAYTAPI_API_TOKEN', '01d70834-def8-41d7-99d8-19a27efbbbb9')
MAYTAPI_PRODUCT_ID = os.getenv('MAYTAPI_PRODUCT_ID', '2ad7cfc1-ad58-4047-ac00-28f6a4074a1a')
MAYTAPI_PHONE_ID = os.getenv('MAYTAPI_PHONE_ID', '86313')

# Validate required environment variables
required_vars = {
    'MAYTAPI_API_TOKEN': MAYTAPI_API_TOKEN,
    'MAYTAPI_PRODUCT_ID': MAYTAPI_PRODUCT_ID,
    'MAYTAPI_PHONE_ID': MAYTAPI_PHONE_ID
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    log_and_print("ERROR", f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

log_and_print("INFO", f"Environment variables loaded. Product ID: {MAYTAPI_PRODUCT_ID}, Phone ID: {MAYTAPI_PHONE_ID}")

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with fixed positioning only"""
        logger.debug(f"Generating master template with product details: {product_details}")
        return f"""Create a stunning, professional marketing poster that looks like it was crafted by an expert graphic designer:

1. SUBJECT TREATMENT (HIGHEST PRIORITY):
   - Keep the original subject 100% UNCHANGED - preserve exact appearance, positioning and all details
   - Do NOT modify, filter, stylize or alter the original subject in ANY way
   - Maintain all original colors, textures, reflections and details of the subject perfectly

2. ELEMENT POSITIONING (MAINTAIN CONSISTENT PLACEMENT):
   - Company name "{product_details.get('company_name', '')}" must be at the TOP of the poster
   - Product name "{product_details.get('product_name', '')}" must be centered ABOVE the product
   - Tagline "{product_details.get('tagline', '')}" must be placed BELOW the product
   - Price "{product_details.get('price', '')}" must be positioned in the BOTTOM-RIGHT corner
   - Address "{product_details.get('address', '')}" must be at the VERY BOTTOM in a single horizontal line

3. DYNAMIC DESIGN ELEMENTS (USE AI REASONING FOR BEST RESULTS):
   - Choose the best fonts, colors, and styling based on the specific product and overall design
   - Determine the optimal size, weight, and visual treatment for each text element
   - Apply professional design techniques that best complement this specific product
   - Use your reasoning to create a cohesive color palette that enhances this particular subject
   - Adapt the background style to suit this specific product while keeping text positions fixed

4. TECHNICAL EXCELLENCE:
   - Maintain exact original aspect ratio
   - Keep safe margins for all text to prevent any edge cropping
   - Ensure perfect readability of all text at various viewing sizes
   - Create pixel-perfect alignment of all design elements

The final result must look like it was created by a professional graphic designer with years of experience in marketing design, with impeccable attention to detail, perfect visual balance, and a high-end aesthetic while keeping the original subject completely untouched.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template with fixed positioning only"""
        logger.debug(f"Generating beverage template with product details: {product_details}")
        return f"""Create a premium café marketing poster for a beverage with these specifications:

1. ELEMENT POSITIONING (MAINTAIN CONSISTENT PLACEMENT):
   - Company name: "{product_details.get('company_name', '')}" must be at the TOP of the poster
   - Product name: "{product_details.get('product_name', '')}" must be placed ABOVE the beverage product
   - Tagline: "{product_details.get('tagline', '')}" must be positioned BELOW the product
   - Price: "{product_details.get('price', '')}" must be in the BOTTOM-RIGHT corner
   - Address: "{product_details.get('address', '')}" must be at the VERY BOTTOM in a single horizontal line

2. PRODUCT ENHANCEMENT:
   - Center the beverage as the focal point
   - Remove the original background completely
   - Keep whipped cream, garnishes, and toppings visible and appetizing
   - Add subtle steam effects for hot drinks or condensation for cold drinks
   - Enhance beverage colors for visual appeal

3. DYNAMIC DESIGN ELEMENTS (USE AI REASONING FOR BEST RESULTS):
   - Use your reasoning to select fonts, colors, and styling that best suit this specific beverage
   - Choose the optimal visual treatment for this particular drink
   - Determine the most appealing background style for this specific product
   - Adapt the design language to enhance this particular beverage's qualities
   - Keep only the positioning fixed while dynamically optimizing all other design aspects

This dynamic approach allows you to create the most appealing design for each unique product while maintaining consistent element positioning across all marketing materials.
"""

    @staticmethod
    def get_food_template(product_details: Dict) -> str:
        """Generate food-specific prompt template with fixed positioning only"""
        logger.debug(f"Generating food template with product details: {product_details}")
        return f"""Create an appetizing marketing poster for a food product with these specifications:

1. ELEMENT POSITIONING (MAINTAIN CONSISTENT PLACEMENT):
   - Company name: "{product_details.get('company_name', '')}" must be at the TOP of the poster
   - Product name: "{product_details.get('product_name', '')}" must be placed ABOVE the food product
   - Tagline: "{product_details.get('tagline', '')}" must be positioned BELOW the product
   - Price: "{product_details.get('price', '')}" must be in the BOTTOM-RIGHT corner
   - Address: "{product_details.get('address', '')}" must be at the VERY BOTTOM in a single horizontal line

2. PRODUCT PRESENTATION:
   - Center the food item as the focal point
   - Remove the original background completely
   - Enhance food textures and colors for appetite appeal
   - Add steam effects for hot items or fresh appearance for cold items
   - Ensure the food looks perfectly prepared

3. DYNAMIC DESIGN ELEMENTS (USE AI REASONING FOR BEST RESULTS):
   - Use your reasoning to select fonts, colors, and styling that best suit this specific food item
   - Choose the optimal visual treatment for this particular dish
   - Determine the most appealing background style for this specific product
   - Adapt the design language to enhance this particular food's qualities
   - Keep only the positioning fixed while dynamically optimizing all other design aspects

This dynamic approach allows you to create the most appealing design for each unique product while maintaining consistent element positioning across all marketing materials.
"""

###################
# IMAGE GENERATION
###################

class ImageGenerator:
    def __init__(self, api_key: str):
        logger.info("Initializing ImageGenerator")
        self.api_key = api_key
        self.client = openai_client
        logger.info(f"ImageGenerator initialized with OpenAI API")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage", logo_image_path: str = None) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Generate a marketing image using OpenAI API with gpt-image-1 model.
        Uses a direct prompt approach with a single API call.
        If logo_image_path is provided, it will be incorporated in the prompt.
        
        Returns:
            Tuple containing (output_path, image_info) where:
            - output_path is the path to the generated image or None on failure
            - image_info is a dictionary with metadata about the image or None on failure
        """
        try:
            start_time = time.time()
            product_name = product_details.get('product_name', 'product')
            
            # Use global logger instead of self.logger
            log_and_print("INFO", f"Generating marketing image for {product_name}")
            
            # Ensure the product image exists
            if not os.path.exists(product_image_path):
                log_and_print("ERROR", f"Product image not found at path: {product_image_path}")
                return None, None
            
            # First, open the image and get dimensions
            original_aspect_ratio = 1.0  # Default value
            with Image.open(product_image_path) as img:
                # Store original dimensions for later use
                original_width, original_height = img.size
                original_aspect_ratio = original_width / original_height
                log_and_print("INFO", f"Original image dimensions: {original_width}x{original_height}, aspect ratio: {original_aspect_ratio:.2f}")
            
            # Now build the enhanced direct prompt with the aspect ratio
            marketing_prompt = f"""
Create a stunning, professional marketing poster that looks like it was crafted by an expert graphic designer:

1. SUBJECT TREATMENT (HIGHEST PRIORITY):
   - Keep the original subject 100% UNCHANGED - preserve exact appearance, positioning and all details
   - Do NOT modify, filter, stylize or alter the original subject in ANY way
   - Maintain all original colors, textures, reflections and details of the subject perfectly

2. EXPERT DESIGN ELEMENTS:
   - Create a visually striking layout with perfect visual hierarchy and balance
   - Design a premium, elegant typography system using complementary professional fonts
   - Position company name "{product_details.get('company_name', '')}" in a commanding position with premium typography
   - Create an eye-catching product name "{product_details.get('product_name', '')}" display with perfect typography
   - Design a professional price badge for "{product_details.get('price', '')}" that draws attention without overwhelming
   - Add tagline "{product_details.get('tagline', '')}" with sophisticated typography that enhances the message
   - Include address "{product_details.get('address', '')}" in a clean, understated way that doesn't distract

3. PROFESSIONAL FINISHES:
   - Create a high-end background that enhances the subject with subtle gradients or textures
   - Apply professional design techniques like layering, subtle shadows, and balanced negative space
   - Use a sophisticated color palette that complements the subject perfectly
   - Add subtle lighting effects that enhance dimensionality without being distracting
   - Implement expert-level visual hierarchy that guides the eye through the composition
   - Ensure perfect alignment of all elements with professional spacing and margins

4. TECHNICAL EXCELLENCE:
   - Maintain exact original aspect ratio ({original_aspect_ratio:.2f})
   - Keep safe margins for all text to prevent any edge cropping
   - Ensure perfect readability of all text at various viewing sizes
   - Create pixel-perfect alignment of all design elements
   - Apply subtle, tasteful design accents that enhance rather than distract

The final result must look like it was created by a professional graphic designer with years of experience in marketing design, with impeccable attention to detail, perfect visual balance, and a high-end aesthetic while keeping the original subject completely untouched.
"""

            # Add product-specific enhancements based on type
            if product_type.lower() == "beverage":
                marketing_prompt += """
5. BEVERAGE-SPECIFIC ENHANCEMENTS:
   - Add subtle visual elements that suggest freshness or temperature (steam for hot drinks, condensation for cold)
   - Use colors and lighting that enhance the beverage's appeal
   - Create an atmosphere that evokes the beverage's qualities (energizing, refreshing, comforting, etc.)
   - Maintain a gourmet/premium aesthetic suitable for a high-end café or restaurant
"""
            elif product_type.lower() == "food":
                marketing_prompt += """
5. FOOD-SPECIFIC ENHANCEMENTS:
   - Add subtle visual elements that enhance appetite appeal
   - Use warm, inviting colors that complement the food
   - Create an elegant, restaurant-quality presentation
   - Maintain a gourmet/premium aesthetic suitable for culinary marketing
"""
            
            # Add logo instruction if provided
            if logo_image_path:
                marketing_prompt += """
6. LOGO PLACEMENT:
   - Integrate the provided logo in a balanced position (typically top-left or top-right)
   - Size the logo appropriately - visible but not overwhelming the composition
   - Ensure the logo stands out clearly against the background
   - Maintain the logo's original appearance and colors
"""
            
            # Add final quality instructions
            marketing_prompt += """
IMPORTANT FINAL INSTRUCTIONS:
- This should look like a professionally designed marketing poster, not an edited photo
- Text should appear as if it was thoughtfully designed and placed by a graphic designer
- The original product must remain COMPLETELY UNALTERED - this is the highest priority
- Ensure all elements work together harmoniously like a cohesive brand design
- Create a result that would be ready for immediate use in professional marketing
"""
            
            log_and_print("INFO", f"Direct marketing prompt: {marketing_prompt[:100]}...")

            # Continue processing the image
            with Image.open(product_image_path) as img:
                # Use 'auto' size parameter to let the model choose the best size
                img_size = "auto"
                log_and_print("INFO", f"Using 'auto' size parameter to let the model choose optimal dimensions")
                
                if img.format != 'PNG':
                    converted_path = f"{os.path.splitext(product_image_path)[0]}_converted.png"
                    img.save(converted_path, format="PNG")
                    product_image_path = converted_path
                    log_and_print("INFO", f"Product image converted to PNG: {product_image_path}")

                file_size = os.path.getsize(product_image_path)
                if file_size > 10 * 1024 * 1024:  # 10MB limit
                    scale_factor = min(1.0, math.sqrt(8 * 1024 * 1024 / file_size))
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    resized_path = f"{os.path.splitext(product_image_path)[0]}_resized.png"
                    img.save(resized_path, format="PNG", optimize=True)
                    product_image_path = resized_path
                    log_and_print("INFO", f"Product image resized to {new_width}x{new_height}")

            # Process logo image if provided
            if logo_image_path and os.path.exists(logo_image_path):
                with Image.open(logo_image_path) as logo_img:
                    if logo_img.format != 'PNG':
                        converted_path = f"{os.path.splitext(logo_image_path)[0]}_converted.png"
                        logo_img.save(converted_path, format="PNG")
                        logo_image_path = converted_path
                        log_and_print("INFO", f"Logo image converted to PNG: {logo_image_path}")

                    file_size = os.path.getsize(logo_image_path)
                    if file_size > 10 * 1024 * 1024:  # 10MB limit
                        scale_factor = min(1.0, math.sqrt(8 * 1024 * 1024 / file_size))
                        new_width = int(logo_img.width * scale_factor)
                        new_height = int(logo_img.height * scale_factor)
                        logo_img = logo_img.resize((new_width, new_height), Image.LANCZOS)
                        resized_path = f"{os.path.splitext(logo_image_path)[0]}_resized.png"
                        logo_img.save(resized_path, format="PNG", optimize=True)
                        logo_image_path = resized_path
                        log_and_print("INFO", f"Logo image resized to {new_width}x{new_height}")

            # Direct editing approach - single API call
            log_and_print("INFO", "Using direct prompt editing approach")
            max_retries = 3
            retry_delay = 2
            result = None
            
            # Use auto quality to let the model choose the best quality
            quality = "auto"  # Using auto instead of low for optimal results
            
            # Keep track of moderation errors to implement fallback strategy
            had_moderation_error = False

            for retry in range(max_retries):
                try:
                    # Use different approach based on whether logo is provided
                    if logo_image_path:
                        log_and_print("INFO", f"Including logo in edit: {logo_image_path}")
                        with open(product_image_path, "rb") as product_file:
                            with open(logo_image_path, "rb") as logo_file:
                                result = self.client.images.edit(
                                    model="gpt-image-1",
                                    prompt=marketing_prompt,
                                    image=[product_file, logo_file],
                                    size=img_size,  # Using determined size based on aspect ratio
                                    quality=quality,
                                    n=1
                                )
                    else:
                        with open(product_image_path, "rb") as product_file:
                            result = self.client.images.edit(
                                model="gpt-image-1",
                                prompt=marketing_prompt,
                                image=product_file,
                                size=img_size,  # Using determined size based on aspect ratio
                                quality=quality,
                                n=1
                            )
                    
                    log_and_print("INFO", "Direct edit API call successful")
                    break
                except Exception as retry_error:
                    error_str = str(retry_error)
                    log_and_print("WARNING", f"API call attempt {retry+1} failed: {error_str}")
                    
                    # Check if this is a safety moderation error
                    if "safety system" in error_str or "moderation_blocked" in error_str:
                        had_moderation_error = True
                        log_and_print("INFO", "Detected safety moderation error, will use fallback prompt")
                        
                        # Create a more generic fallback prompt that avoids moderation issues
                        # This typically happens with images containing people, copyrighted content, etc.
                        fallback_prompt = f"""
Create a clean, professional marketing design with the following elements:
- Preserve the main subject completely intact, maintaining all details and characteristics
- Add a subtle, elegant background that complements the subject
- Include text for company name "{product_details.get('company_name', '')}" in a premium font
- Add text for product name "{product_details.get('product_name', '')}" in an attractive font
- Include price "{product_details.get('price', '')}" in a small badge design
- Add tagline "{product_details.get('tagline', '')}" in a complementary style
- Include address "{product_details.get('address', '')}" in small text at the bottom

Design this as a clean, minimalist but professional marketing poster with balanced layout.
Ensure all text is placed within safe margins away from edges to prevent cropping.
"""
                        # Update the marketing prompt for the next retry
                        marketing_prompt = fallback_prompt
                    
                    if retry < max_retries - 1:
                        log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        log_and_print("ERROR", "All API call retries failed")
                        return None, None

            # Process the result
            if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                b64_data = result.data[0].b64_json
                image_bytes = base64.b64decode(b64_data)
                
                # Read the generated image to check dimensions
                try:
                    with Image.open(BytesIO(image_bytes)) as generated_img:
                        generated_width, generated_height = generated_img.size
                        generated_format = generated_img.format
                        generated_mode = generated_img.mode
                        
                        log_and_print("INFO", f"Raw OpenAI image properties:")
                        log_and_print("INFO", f"- Dimensions: {generated_width}x{generated_height}")
                        log_and_print("INFO", f"- Format: {generated_format}")
                        log_and_print("INFO", f"- Mode: {generated_mode}")
                        log_and_print("INFO", f"- Size: {len(image_bytes)} bytes")
                        
                        # Check if aspect ratio is significantly different from original
                        if original_width > 0 and original_height > 0:
                            generated_aspect = generated_width / generated_height
                            log_and_print("INFO", f"Generated aspect ratio: {generated_aspect:.2f} vs Original: {original_aspect_ratio:.2f}")
                            
                            # If aspect ratios are very different (tolerance of 0.1), log a warning
                            if abs(generated_aspect - original_aspect_ratio) > 0.1:
                                log_and_print("WARNING", f"Aspect ratio changed from {original_aspect_ratio:.2f} to {generated_aspect:.2f}")
                except Exception as img_err:
                    log_and_print("WARNING", f"Could not analyze generated image: {str(img_err)}")
            else:
                log_and_print("ERROR", "No image data in response")
                return None, None
                
            # Save the image directly from OpenAI response bytes without additional processing
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            product_name_safe = ''.join(c if c.isalnum() else '_' for c in product_name)[:20]
            output_filename = f"{product_name_safe}_{timestamp}.png"
            output_path = os.path.join("images/output", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(image_bytes)
            log_and_print("INFO", f"Marketing image saved to {output_path} without additional processing")
            
            # Try to upload to Imgur for a public link
            public_url = None
            if USE_IMGUR:
                try:
                    public_url = upload_image_to_imgur(image_bytes)
                    if public_url:
                        log_and_print("INFO", f"Public image URL: {public_url}")
                except Exception as imgur_err:
                    log_and_print("ERROR", f"Error uploading to Imgur: {str(imgur_err)}")
            
            # Generate local URL for the image
            app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'https://auto-design-production.up.railway.app'))
            local_url = f"{app_url}/images/output/{os.path.basename(output_path)}"
            
            # Store URLs in image metadata
            image_info = {
                "local_url": local_url,
                "public_url": public_url,
                "path": output_path,
                "timestamp": timestamp
            }

            processing_time = time.time() - start_time
            log_and_print("INFO", f"Total processing time: {processing_time:.2f} seconds")
            return output_path, image_info
                
        except Exception as e:
            log_and_print("ERROR", f"Error generating marketing image: {str(e)}")
            log_and_print("ERROR", f"Traceback: {traceback.format_exc()}")
            return None, None

###################
# MAYTAPI WHATSAPP API
###################

class MaytapiClient:
    def __init__(self, product_id: str, api_token: str, phone_id: str):
        log_and_print("INFO", f"Initializing MaytapiClient with product ID: {product_id}, phone ID: {phone_id}")
        self.product_id = product_id
        self.api_token = api_token
        self.phone_id = phone_id
        self.api_base_url = f"https://api.maytapi.com/api/{product_id}"
        self.headers = {
            "x-maytapi-key": self.api_token,
            "Content-Type": "application/json"
        }
        log_and_print("INFO", "MaytapiClient initialized successfully")
        
        # Test connection to ensure credentials are valid
        self._test_connection()
    
    def _test_connection(self):
        """Test the API connection to ensure credentials are valid"""
        try:
            # Checking phone status is a simple API call to verify credentials
            response = self._make_request("GET", f"/{self.phone_id}/status")
            if response.get("success"):
                log_and_print("INFO", "Maytapi connection test successful")
                print(f"[DEBUG] Phone status: {json.dumps(response)}")
            else:
                log_and_print("WARNING", f"Maytapi connection test returned unsuccessful: {json.dumps(response)}")
        except Exception as e:
            log_and_print("WARNING", f"Maytapi connection test failed: {str(e)}")
            # We'll continue anyway, as this is just a test
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make a request to the Maytapi API with better error handling"""
        url = f"{self.api_base_url}{endpoint}"
        log_and_print("INFO", f"Making {method} request to {url}")
        print(f"[DEBUG] Headers: {json.dumps(self.headers)}")
        if data:
            print(f"[DEBUG] Request data: {json.dumps(data)}")
        
        max_retries = 3
        retry_delay = 1
        
        for retry in range(max_retries):
            try:
                if method.upper() == "GET":
                    response = requests.get(url, headers=self.headers, timeout=30)
                elif method.upper() == "POST":
                    response = requests.post(url, headers=self.headers, json=data, timeout=30)
                else:
                    log_and_print("ERROR", f"Invalid method: {method}")
                    return {"success": False, "error": "Invalid method"}
                
                # Log response status
                log_and_print("INFO", f"Response status code: {response.status_code}")
                
                # Try to get JSON response
                try:
                    result = response.json()
                    if response.status_code >= 400:
                        log_and_print("ERROR", f"API error: HTTP {response.status_code}")
                        if retry < max_retries - 1:
                            log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                    
                    # Success - return result
                    return result
                except Exception as json_error:
                    log_and_print("ERROR", f"Failed to parse JSON response: {str(json_error)}")
                    # Check if we should retry
                    if retry < max_retries - 1:
                        log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    return {"success": False, "error": f"Invalid JSON response: {response.text[:200]}..."}
                    
            except requests.exceptions.Timeout:
                log_and_print("ERROR", "Request timeout to Maytapi API")
                if retry < max_retries - 1:
                    log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return {"success": False, "error": "API request timed out"}
                
            except requests.exceptions.ConnectionError:
                log_and_print("ERROR", "Connection error to Maytapi API")
                if retry < max_retries - 1:
                    log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return {"success": False, "error": "API connection error"}
                
            except Exception as e:
                log_and_print("ERROR", f"Error making request to Maytapi API: {str(e)}")
                if retry < max_retries - 1:
                    log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                return {"success": False, "error": str(e)}
        
        # If we get here, all retries have failed
        return {"success": False, "error": "All retries failed"}
    
    def get_media_content(self, message_id: str) -> Optional[bytes]:
        """
        Fetches media content from Maytapi API.
        
        Args:
            message_id: The ID of the message containing the media
            
        Returns:
            Optional[bytes]: The media content as bytes if successful, None otherwise
        """
        try:
            # First, get the message details to check if it's a media message
            message_endpoint = f"/{self.phone_id}/getMessage/{message_id}"
            response = self._make_request("GET", message_endpoint)
            
            if not response.get('success'):
                log_and_print("ERROR", f"Failed to get message details: {response}")
                return None
                
            message_data = response.get('data', [])
            if not message_data:
                log_and_print("ERROR", "No message data found")
                return None
                
            # Get the first message
            message = message_data[0].get('message', {})
            message_type = message.get('type', '')
            
            if message_type not in ['image', 'document', 'video', 'audio']:
                log_and_print("ERROR", f"Message is not a media type: {message_type}")
                return None
                
            # Check for media URL in different possible locations
            media_url = None
            
            # Try direct media field first
            if 'media' in message:
                media_url = message['media']
                log_and_print("INFO", f"Found media URL in message.media: {media_url[:50]}...")
            
            # Try message.url field (which seems to be used in some cases)
            elif 'url' in message:
                media_url = message['url']
                log_and_print("INFO", f"Found media URL in message.url: {media_url[:50]}...")
            
            # Try body field for base64 data
            elif 'body' in message and isinstance(message['body'], str) and message['body'].startswith('data:'):
                try:
                    # Extract base64 data from data URL
                    base64_data = message['body'].split(',')[1]
                    return base64.b64decode(base64_data)
                except Exception as e:
                    log_and_print("ERROR", f"Failed to decode base64 data: {str(e)}")
                    return None
            
            if not media_url:
                log_and_print("ERROR", "No media URL found in message")
                return None
                
            # Fetch the media content
            try:
                response = requests.get(
                    media_url,
                    headers={
                        'x-maytapi-key': self.api_token,
                        'Content-Type': 'application/json'
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Verify it's actually image data
                    try:
                        Image.open(BytesIO(response.content))
                        return response.content
                    except Exception as img_err:
                        log_and_print("ERROR", f"Downloaded data is not a valid image: {str(img_err)}")
                        return None
                else:
                    log_and_print("ERROR", f"Failed to fetch media: HTTP {response.status_code}")
                    return None
                    
            except requests.exceptions.RequestException as e:
                log_and_print("ERROR", f"Error fetching media: {str(e)}")
                return None
                
        except Exception as e:
            log_and_print("ERROR", f"Error in get_media_content: {str(e)}")
            return None
    
    def send_message(self, to_number: str, message: str, typing: bool = True) -> Dict:
        """Send a text message via WhatsApp with improved rate limiting per user
        
        Args:
            to_number: The recipient's phone number (with country code, no special chars)
            message: The text message to send
            typing: Whether to show typing indicator before sending
            
        Returns:
            Dict with success status and data
        """
        # Apply rate limiting PER USER
        current_time = time.time()
        if to_number in last_message_time and current_time - last_message_time[to_number] < 1:
            delay_time = 1 - (current_time - last_message_time[to_number])
            log_and_print("INFO", f"Rate limiting message to {to_number} - waiting {delay_time:.2f} seconds")
            time.sleep(delay_time)
        
        # Clean the phone number - only digits, no special chars
        to_number = self._clean_phone_number(to_number)
        
        data = {
            "to_number": to_number,
            "type": "text",
            "message": message
        }
        
        # Add typing indicator if requested
        if typing:
            data["typing"] = "typing"
            # Scale typing duration based on message length (1 sec minimum, 5 sec maximum)
            typing_duration = min(5, max(1, len(message) / 100))
            data["duration"] = typing_duration
        
        # Log only the first 50 characters of the message to avoid log clutter
        truncated_message = message[:50] + ('...' if len(message) > 50 else '')
        log_and_print("INFO", f"Sending message to {to_number}: {truncated_message}")
        
        result = self._make_request("POST", f"/{self.phone_id}/sendMessage", data)
        
        if result.get("success"):
            # Update rate limiting timestamp for this specific user
            last_message_time[to_number] = time.time()
            log_and_print("INFO", f"Message sent successfully to {to_number}")
            return {"success": True, "data": result.get("data", {})}
        else:
            error_msg = result.get("error", "Unknown error")
            log_and_print("ERROR", f"Error sending message to {to_number}: {error_msg}")
            return {"success": False, "error": error_msg}
    
    def send_media(self, to_number: str, caption: str = "", media_url: str = None, 
                   media_base64: str = None, filename: str = None) -> Dict:
        """Send a media message via WhatsApp with improved reliability
        
        Args:
            to_number: The recipient's phone number
            caption: Optional text caption for the media
            media_url: URL of the media (either this or media_base64 must be provided)
            media_base64: Base64-encoded media data
            filename: Filename for the media (required for base64)
            
        Returns:
            Dict with success status and data
        """
        # Apply rate limiting PER USER
        current_time = time.time()
        if to_number in last_message_time and current_time - last_message_time[to_number] < 2:
            delay_time = 2 - (current_time - last_message_time[to_number])
            log_and_print("INFO", f"Rate limiting media message to {to_number} - waiting {delay_time:.2f} seconds")
            time.sleep(delay_time)
        
        # Clean the phone number - only digits, no special chars
        to_number = self._clean_phone_number(to_number)
        
        data = {
            "to_number": to_number,
            "type": "media",
            "text": caption  # Maytapi uses 'text' for caption
        }
        
        # Add either URL or base64 data
        if media_url:
            log_and_print("INFO", f"Sending media from URL to {to_number}")
            data["message"] = media_url
        elif media_base64:
            # Determine image format from filename or default to png
            image_format = "png"
            if filename and '.' in filename:
                ext = filename.split('.')[-1].lower()
                if ext in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                    image_format = ext
                    if image_format == 'jpeg':
                        image_format = 'jpg'  # standardize jpeg to jpg
            
            log_and_print("INFO", f"Sending media from base64 to {to_number}, format: {image_format}, base64 length: {len(media_base64)} chars")
            data["message"] = f"data:image/{image_format};base64,{media_base64}"
            
            # Set filename if provided
            if filename:
                data["filename"] = filename
        else:
            log_and_print("ERROR", "No media URL or base64 data provided")
            return {"success": False, "error": "No media provided"}
        
        # Add retry logic for media sending
        max_retries = 3
        retry_delay = 2
        
        for retry in range(max_retries):
            log_and_print("INFO", f"Media send attempt {retry+1} to {to_number}")
            result = self._make_request("POST", f"/{self.phone_id}/sendMessage", data)
            
            if result.get("success"):
                # Update rate limiting timestamp for this specific user
                last_message_time[to_number] = time.time()
                log_and_print("INFO", f"Media sent successfully to {to_number}")
            return {"success": True, "data": result.get("data", {})}
        else:
                error_msg = result.get("error", "Unknown error")
                log_and_print("ERROR", f"Error sending media (attempt {retry+1}): {error_msg}")
                
                # Check for specific error types
                if "too large" in error_msg.lower() or "file size" in error_msg.lower():
                    log_and_print("ERROR", "Media file too large")
                    return {"success": False, "error": "Media file too large"}
                
                # If not the last retry, wait and try again
                if retry < max_retries - 1:
                    log_and_print("INFO", f"Retrying media send in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    log_and_print("ERROR", "All media send retries failed")
                    return {"success": False, "error": error_msg}
        
        # If we get here, all retries failed
        return {"success": False, "error": "Failed to send media after multiple attempts"}
    
    def _clean_phone_number(self, phone_number: str) -> str:
        """Clean a phone number to contain only digits with country code
        
        Args:
            phone_number: The phone number to clean
            
        Returns:
            Cleaned phone number with only digits, preserving country code
        """
        # If already in Maytapi format, return as is
        if '@c.us' in phone_number:
            # Extract just the numbers from Maytapi format
            return phone_number.split('@')[0]
            
        # Remove any non-numeric characters
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Ensure it has country code (this is simplified, might need more sophistication)
        if len(clean_number) < 10:
            log_and_print("WARNING", f"Phone number too short, might be missing country code: {clean_number}")
        
        return clean_number
    
    def check_phone_number(self, phone_number: str) -> bool:
        """Check if a phone number is registered on WhatsApp
        
        Args:
            phone_number: The phone number to check
            
        Returns:
            Boolean indicating if the number is registered
        """
        # Clean the phone number
        clean_number = self._clean_phone_number(phone_number)
        
        # Use the checkPhones endpoint
        data = {
            "numbers": [clean_number]
        }
        
        result = self._make_request("POST", f"/{self.phone_id}/checkPhones", data)
        
        if result.get("success") and "data" in result:
            # Check the first number's validity
            numbers_data = result["data"]
            if numbers_data and len(numbers_data) > 0:
                is_valid = numbers_data[0].get("valid", False)
                log_and_print("INFO", f"Phone number {clean_number} is valid on WhatsApp: {is_valid}")
                return is_valid
        
        log_and_print("WARNING", f"Failed to check if phone number is valid: {result.get('error', 'Unknown error')}")
        # Default to True to prevent blocking legitimate numbers if check fails
        return True

###################
# SESSION MANAGER
###################

class SessionManager:
    """Manages user sessions with proper isolation"""
    
    def __init__(self):
        self.sessions = {}
        # Keep track of which users have been welcomed globally
        self.welcomed_users = set()
        log_and_print("INFO", "Session Manager initialized")
    
    def get_user_id(self, from_number: str) -> str:
        """Generate a consistent unique user ID from phone number
        
        Args:
            from_number: User's WhatsApp number (e.g. 1234567890@c.us)
            
        Returns:
            Unique user ID
        """
        # Extract just the number part if Maytapi format
        if '@' in from_number:
            number = from_number.split('@')[0]
        else:
            number = from_number
            
        # Clean the number to just digits
        number = ''.join(filter(str.isdigit, number))
        
        # Hash it for consistent unique ID
        return hashlib.md5(number.encode()).hexdigest()
    
    def create_session(self, from_number: str) -> Dict:
        """Create a new session for a user
        
        Args:
            from_number: User's WhatsApp number
            
        Returns:
            New session dictionary
        """
        user_id = self.get_user_id(from_number)
        
        # Check if this user has been welcomed before
        is_welcomed = user_id in self.welcomed_users
        
        # Create a new isolated session
        session = {
            "user_id": user_id,
            "phone_number": from_number,
            "state": "waiting_for_command",
            "product_image": None,
            "details": {},
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "welcomed": is_welcomed  # Track welcomed status
        }
        
        # Store in sessions dictionary, keyed by user_id for isolation
        self.sessions[user_id] = session
        log_and_print("INFO", f"Created new session for user {user_id}, welcomed={is_welcomed}")
        return session
    
    def get_session(self, from_number: str, create_if_not_exists: bool = True) -> Dict:
        """Get a user's session
        
        Args:
            from_number: User's WhatsApp number
            create_if_not_exists: Whether to create a session if none exists
            
        Returns:
            User's session dictionary
        """
        user_id = self.get_user_id(from_number)
        
        # Check if session exists
        if user_id in self.sessions:
            # Update last active time
            self.sessions[user_id]["last_active"] = datetime.now().isoformat()
            log_and_print("DEBUG", f"Retrieved existing session for user {user_id}")
            return self.sessions[user_id]
        
        # Create new session if requested
        if create_if_not_exists:
            return self.create_session(from_number)
        
        log_and_print("DEBUG", f"No session found for user {user_id} and not creating one")
        return None
    
    def update_session(self, from_number: str, update_data: Dict) -> Dict:
        """Update a user's session with new data
        
        Args:
            from_number: User's WhatsApp number
            update_data: Dictionary of data to update
            
        Returns:
            Updated session dictionary
        """
        # Get user_id for isolation
        user_id = self.get_user_id(from_number)
        
        # Ensure session exists
        if user_id not in self.sessions:
            log_and_print("WARNING", f"Attempted to update non-existent session for {user_id}")
            return self.create_session(from_number)
        
        # Get the current session
        session = self.sessions[user_id]
        
        # Update with new data
        for key, value in update_data.items():
            session[key] = value
            
            # Special handling for welcomed status
            if key == 'welcomed' and value == True:
                self.welcomed_users.add(user_id)
                log_and_print("INFO", f"Added user {user_id} to welcomed users list")
        
        # Update last active time
        session["last_active"] = datetime.now().isoformat()
        
        # Store back to sessions
        self.sessions[user_id] = session
        
        log_and_print("INFO", f"Updated session for user {user_id} with keys: {', '.join(update_data.keys())}")
        return session
    
    def end_session(self, from_number: str) -> bool:
        """End a user's session by resetting to initial state
        
        Args:
            from_number: User's WhatsApp number
            
        Returns:
            Boolean indicating if session was ended
        """
        user_id = self.get_user_id(from_number)
        
        if user_id in self.sessions:
            # Get current welcomed state
            is_welcomed = user_id in self.welcomed_users
            
            # Reset session to initial state, but keep welcomed state
            self.sessions[user_id] = {
                "user_id": user_id,
                "phone_number": from_number,
                "state": "waiting_for_command",
                "product_image": None,
                "details": {},
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat(),
                "welcomed": is_welcomed  # Preserve welcomed state
            }
            log_and_print("INFO", f"Ended session for user {user_id}, keeping welcomed state: {is_welcomed}")
            return True
        
        log_and_print("WARNING", f"Attempted to end non-existent session for {user_id}")
        return False
    
    def set_welcomed(self, from_number: str, welcomed: bool = True) -> None:
        """Set a user as welcomed in the registry
        
        Args:
            from_number: User's WhatsApp number
            welcomed: Whether the user has been welcomed
        """
        user_id = self.get_user_id(from_number)
        
        if welcomed:
            self.welcomed_users.add(user_id)
            log_and_print("INFO", f"Marked user {user_id} as welcomed")
        elif user_id in self.welcomed_users:
            self.welcomed_users.remove(user_id)
            log_and_print("INFO", f"Removed welcomed status for user {user_id}")
        
        # Also update in current session if it exists
        if user_id in self.sessions:
            self.sessions[user_id]['welcomed'] = welcomed
            log_and_print("INFO", f"Updated welcomed status in session for user {user_id}: {welcomed}")
    
    def clean_old_sessions(self, max_age_hours: int = 24) -> int:
        """Clean up sessions older than specified hours
        
        Args:
            max_age_hours: Maximum session age in hours
            
        Returns:
            Number of sessions cleaned
        """
        now = datetime.now()
        sessions_to_remove = []
        
        for user_id, session in self.sessions.items():
            last_active = datetime.fromisoformat(session["last_active"])
            age_hours = (now - last_active).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                sessions_to_remove.append(user_id)
        
        # Remove the old sessions
        for user_id in sessions_to_remove:
            del self.sessions[user_id]
        
        if sessions_to_remove:
            log_and_print("INFO", f"Cleaned {len(sessions_to_remove)} old sessions")
        
        return len(sessions_to_remove)
    
    def get_all_sessions(self) -> Dict[str, Dict]:
        """Get all active sessions (primarily for debugging)
        
        Returns:
            Dictionary of all sessions
        """
        return self.sessions

###################
# IMAGE HANDLER
###################

class ImageHandler:
    """Handles image processing and storage with improved reliability"""
    
    def __init__(self, maytapi_client: MaytapiClient):
        """Initialize with MaytapiClient for direct image fetching"""
        self.maytapi_client = maytapi_client
        log_and_print("INFO", "ImageHandler initialized")
    
    def save_image_from_bytes(self, image_bytes: bytes, user_id: str, filename: str = None) -> Tuple[str, bool]:
        """Save image bytes to disk with validation and error handling
        
        Args:
            image_bytes: Raw image data
            user_id: Unique user ID
            filename: Optional filename
            
        Returns:
            Tuple of (file path, success boolean)
        """
        try:
            # Validate image data
            if not image_bytes or len(image_bytes) < 100:
                log_and_print("ERROR", f"Invalid image data: too small ({len(image_bytes) if image_bytes else 0} bytes)")
                return None, False
            
            # Verify it's valid image data
            try:
                img = Image.open(BytesIO(image_bytes))
                img_format = img.format
                img_size = img.size
                log_and_print("INFO", f"Valid image detected in bytes: {img_format}, {img_size}px")
            except Exception as img_error:
                log_and_print("ERROR", f"Invalid image data: {str(img_error)}")
                return None, False
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # Use image format as extension if detected
                extension = img_format.lower() if img_format else "jpg"
                filename = f"user_{user_id}_{timestamp}.{extension}"
            
            # Ensure input directory exists
            os.makedirs("images/input", exist_ok=True)
            
            # Full path to save image
            image_path = os.path.join("images/input", filename)
            
            # Save bytes directly
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Verify file was saved
            if not os.path.exists(image_path):
                log_and_print("ERROR", f"Failed to save image to {image_path}")
                return None, False
            
            # Verify file size
            file_size = os.path.getsize(image_path)
            if file_size <= 0:
                log_and_print("ERROR", f"Saved image file is empty: {image_path}")
                os.remove(image_path)
                return None, False
            elif file_size > 20 * 1024 * 1024:  # 20MB limit
                log_and_print("WARNING", f"Image file is very large ({file_size} bytes), attempting to resize")
                
                # Resize the image to reduce file size
                try:
                    with Image.open(image_path) as img:
                        # Calculate scaling factor to get under 10MB
                        scale_factor = (10 * 1024 * 1024 / file_size) ** 0.5
                        new_width = int(img.width * scale_factor)
                        new_height = int(img.height * scale_factor)
                        
                        # Only resize if it's actually making the image smaller
                        if scale_factor < 1.0:
                            log_and_print("INFO", f"Resizing image from {img.size} to {new_width}x{new_height}")
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                            img.save(image_path, optimize=True)
                            log_and_print("INFO", f"Image resized and saved, new size: {os.path.getsize(image_path)} bytes")
                except Exception as resize_error:
                    log_and_print("ERROR", f"Failed to resize large image: {str(resize_error)}")
                    # Continue with the original image since we at least saved it
            
            log_and_print("INFO", f"Image saved to {image_path} ({file_size} bytes)")
            return image_path, True
                
        except Exception as e:
            log_and_print("ERROR", f"Error saving image: {str(e)}")
            traceback.print_exc()
            return None, False
    
    def extract_image_from_maytapi(self, webhook_data: Dict, from_number: str) -> Optional[bytes]:
        """Extract image bytes from Maytapi webhook data with improved reliability
        
        Args:
            webhook_data: Full webhook data from Maytapi
            from_number: User's WhatsApp number (for logging)
            
        Returns:
            Image bytes or None if extraction failed
        """
        try:
            user_id = hashlib.md5(from_number.encode()).hexdigest()[:8]  # Short ID for logging
            log_and_print("INFO", f"Attempting to extract image from webhook data for user {user_id}")
            
            # STRATEGY 1: Direct message ID fetch
            message_data = webhook_data.get('message', {})
            message_id = message_data.get('id')
            
            if message_id:
                log_and_print("INFO", f"Found message ID: {message_id}, attempting direct media fetch")
                image_bytes = self.maytapi_client.get_media_content(message_id)
                if image_bytes and len(image_bytes) > 100:
                    log_and_print("INFO", f"Successfully fetched image directly using message ID: {len(image_bytes)} bytes")
                    return image_bytes
                else:
                    log_and_print("WARNING", "Direct message ID fetch failed or returned invalid data, trying alternative methods")
            
            # STRATEGY 2: Check for direct media URL in the message object
            if isinstance(message_data, dict):
                log_and_print("INFO", "Checking for media URL in message data")
                
                # Try standard media field
                if 'media' in message_data and message_data['media']:
                    media_url = message_data['media']
                    log_and_print("INFO", f"Found media URL in message: {media_url[:50]}...")
                    try:
                        response = requests.get(media_url, timeout=30)
                        if response.status_code == 200 and len(response.content) > 100:
                            log_and_print("INFO", f"Successfully downloaded image from media URL: {len(response.content)} bytes")
                            return response.content
                    except Exception as url_error:
                        log_and_print("ERROR", f"Failed to download from media URL: {str(url_error)}")
                
                # Try different possible fields where media might be stored
                for field in ['file', 'thumbnail', 'mediaUrl']:
                    if field in message_data and message_data[field]:
                        media_url = message_data[field]
                        if isinstance(media_url, str) and (media_url.startswith('http') or media_url.startswith('data:')):
                            log_and_print("INFO", f"Found alternative media URL in '{field}': {media_url[:50]}...")
                            
                            # Handle data URI format
                            if media_url.startswith('data:'):
                                try:
                                    content_type, b64data = media_url.split(',', 1)
                                    log_and_print("INFO", f"Extracting base64 data from data URI")
                                    decoded_data = base64.b64decode(b64data)
                                    if len(decoded_data) > 100:
                                        return decoded_data
                                    else:
                                        log_and_print("ERROR", f"Decoded data too small: {len(decoded_data)} bytes")
                                except Exception as b64_error:
                                    log_and_print("ERROR", f"Failed to decode data URI: {str(b64_error)}")
                            
                            # Handle HTTP URL format
                            elif media_url.startswith('http'):
                                try:
                                    response = requests.get(media_url, timeout=30)
                                    if response.status_code == 200 and len(response.content) > 100:
                                        log_and_print("INFO", f"Successfully downloaded image from {field} URL: {len(response.content)} bytes")
                                        return response.content
                                except Exception as url_error:
                                    log_and_print("ERROR", f"Failed to download from {field} URL: {str(url_error)}")
            
            # STRATEGY 3: Look for body field with base64 data
            if isinstance(message_data, dict) and 'body' in message_data and message_data['body']:
                try:
                    log_and_print("INFO", "Trying to decode base64 data from body field")
                    body_data = message_data['body']
                    if isinstance(body_data, str):
                        # If it's base64 encoded
                        if body_data.startswith('data:'):
                            content_type, b64data = body_data.split(',', 1)
                            decoded_data = base64.b64decode(b64data)
                            if len(decoded_data) > 100:
                                return decoded_data
                        else:
                            # Sometimes it's just base64 without the prefix
                            try:
                                decoded_data = base64.b64decode(body_data)
                                if len(decoded_data) > 100:
                                    # Verify it's an image by trying to open it
                                    try:
                                        Image.open(BytesIO(decoded_data))
                                        return decoded_data
                                    except:
                                        pass
                            except:
                                pass
                except Exception as body_error:
                    log_and_print("ERROR", f"Failed to extract image from body: {str(body_error)}")
            
            # STRATEGY 4: Deep search in all webhook data
            log_and_print("INFO", "Deep searching webhook data for image data")
            
            def search_dict_for_image(data, path=""):
                if not isinstance(data, dict):
                    return None
                
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Check if this looks like a media URL
                    if isinstance(value, str) and key in ['file', 'media', 'url', 'mediaUrl', 'thumbnail', 'image']:
                        if value.startswith('http'):
                            log_and_print("INFO", f"Found potential media URL in {current_path}: {value[:50]}...")
                            try:
                                response = requests.get(value, timeout=30)
                                if response.status_code == 200 and len(response.content) > 100:
                                    # Try to open as image to verify
                                    try:
                                        Image.open(BytesIO(response.content))
                                        log_and_print("INFO", f"Successfully downloaded verified image from {current_path}: {len(response.content)} bytes")
                                        return response.content
                                    except:
                                        log_and_print("WARNING", f"Downloaded content from {current_path} is not a valid image")
                            except Exception as search_error:
                                log_and_print("ERROR", f"Failed to download from {current_path}: {str(search_error)}")
                        
                        elif value.startswith('data:'):
                            log_and_print("INFO", f"Found potential data URI in {current_path}")
                            try:
                                content_type, b64data = value.split(',', 1)
                                image_data = base64.b64decode(b64data)
                                if image_data and len(image_data) > 100:
                                    # Try to open as image to verify
                                    try:
                                        Image.open(BytesIO(image_data))
                                        log_and_print("INFO", f"Successfully decoded verified image from {current_path}: {len(image_data)} bytes")
                                        return image_data
                                    except:
                                        log_and_print("WARNING", f"Decoded content from {current_path} is not a valid image")
                            except Exception as data_error:
                                log_and_print("ERROR", f"Failed to decode from {current_path}: {str(data_error)}")
                    
                    # Recurse into nested dictionaries
                    elif isinstance(value, dict):
                        result = search_dict_for_image(value, current_path)
                        if result:
                            return result
                    
                    # Check lists too
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                result = search_dict_for_image(item, f"{current_path}[{i}]")
                                if result:
                                    return result
                
                return None
            
            # Try deep search
            image_data = search_dict_for_image(webhook_data)
            if image_data:
                return image_data
            
            # No image data found
            log_and_print("ERROR", f"Failed to extract image data for user {user_id} using all available methods")
            return None
            
        except Exception as e:
            log_and_print("ERROR", f"Error in extract_image_from_maytapi: {str(e)}")
            traceback.print_exc()
            return None
    
    def create_placeholder_image(self, message: str = "Placeholder Image") -> bytes:
        """Create a simple placeholder image with message
        
        Args:
            message: Message to display on the placeholder
            
        Returns:
            Image bytes for a placeholder
        """
        try:
            # Create a simple white 600x600 image
            img = Image.new('RGB', (600, 600), color='white')
            
            # Add some text indicating it's a placeholder
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Try to use a system font, fall back to default if not available
            try:
                # Try common fonts that might be available
                for font_name in ['Arial', 'DejaVuSans', 'FreeSans', 'Verdana', 'Tahoma']:
                    try:
                        font = ImageFont.truetype(font_name, 30)
                        break
                    except:
                        continue
                else:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Draw main message
            text = message
            
            # Handle different PIL versions for text drawing
            try:
                # For newer PIL versions
                text_width = draw.textlength(text, font=font)
                draw.text(
                    ((600 - text_width) / 2, 250),
                    text,
                    fill='black',
                    font=font
                )
                
                # Add instructions
                instructions = "Please try sending your image again"
                inst_width = draw.textlength(instructions, font=font)
                draw.text(
                    ((600 - inst_width) / 2, 300),
                    instructions,
                    fill='black',
                    font=font
                )
                
            except AttributeError:
                # For older PIL versions
                text_width, _ = draw.textsize(text, font=font)
                draw.text(
                    ((600 - text_width) / 2, 250),
                    text,
                    fill='black',
                    font=font
                )
                
                # Add instructions
                instructions = "Please try sending your image again"
                inst_width, _ = draw.textsize(instructions, font=font)
                draw.text(
                    ((600 - inst_width) / 2, 300),
                    instructions,
                    fill='black',
                    font=font
                )
            
            # Add a border
            draw.rectangle([(20, 20), (580, 580)], outline='black', width=2)
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            log_and_print("INFO", "Created placeholder image")
            return img_byte_arr.getvalue()
            
        except Exception as e:
            log_and_print("ERROR", f"Error creating placeholder image: {str(e)}")
            # Return a minimal 1x1 transparent PNG as fallback
            return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, maytapi_product_id: str, maytapi_api_token: str, maytapi_phone_id: str):
        log_and_print("INFO", "Initializing MarketingBot")
        self.image_generator = ImageGenerator(openai_key)
        self.whatsapp_client = MaytapiClient(
            product_id=maytapi_product_id,
            api_token=maytapi_api_token,
            phone_id=maytapi_phone_id
        )
        self.session_manager = SessionManager()
        self.image_handler = ImageHandler(self.whatsapp_client)
        log_and_print("INFO", "MarketingBot initialized with all components")
    
    def process_request(self, user_id: str, from_number: str, product_image_path: str, product_details: Dict, product_type: str = "beverage", share_link_only: bool = False) -> Dict:
        """Process a marketing image request with improved error handling
        
        Args:
            user_id: Unique user ID
            from_number: User's WhatsApp number
            product_image_path: Path to product image
            product_details: Dictionary of product details
            product_type: Type of product (beverage, food, etc.)
            share_link_only: If True, share a link to the image instead of the image itself
            
        Returns:
            Dictionary with process results
        """
        try:
            log_and_print("INFO", f"Processing request for user {user_id}")
            print(f"[DEBUG] Product details: {json.dumps(product_details)}")
            print(f"[DEBUG] Product image path: {product_image_path}, exists: {os.path.exists(product_image_path)}")
            
            # Verify image exists
            if not product_image_path or not os.path.exists(product_image_path):
                log_and_print("ERROR", f"Product image does not exist: {product_image_path}")
                
                self.whatsapp_client.send_message(
                    from_number,
                    "❌ Error: The product image was not found.\n\n"
                    "Please try again by sending 'edit' and then uploading a new image."
                )
                return {
                    "success": False,
                    "error": "Product image not found"
                }
            
            # Verify image file is valid
            try:
                with Image.open(product_image_path) as img:
                    img_format = img.format
                    img_size = img.size
                    log_and_print("INFO", f"Verified image is valid: {img_format}, {img_size}px")
            except Exception as img_error:
                log_and_print("ERROR", f"Invalid image file: {str(img_error)}")
                
                self.whatsapp_client.send_message(
                    from_number,
                    "❌ Error: The product image file is invalid or corrupted.\n\n"
                    "Please try again by sending 'edit' and then uploading a new image."
                )
                return {
                    "success": False,
                    "error": f"Invalid image file: {str(img_error)}"
                }
            
            # Verify required product details
            required_fields = ['company_name', 'product_name', 'price']
            missing_fields = [field for field in required_fields if not product_details.get(field)]
            
            if missing_fields:
                log_and_print("ERROR", f"Missing required product details: {', '.join(missing_fields)}")
                
                self.whatsapp_client.send_message(
                    from_number,
                    f"❌ Error: Missing required product details: {', '.join(missing_fields)}.\n\n"
                    "Please provide all required information and try again."
                )
                return {
                    "success": False,
                    "error": f"Missing product details: {', '.join(missing_fields)}"
                }
            
            # Send processing message to user
            self.whatsapp_client.send_message(
                from_number,
                "✨ Generating your marketing image...\n"
                "This may take up to 30 seconds."
            )
            
            # Generate marketing image
            log_and_print("INFO", f"Starting image generation for user {user_id}")
            output_path, image_info = self.image_generator.generate_marketing_image(
                product_image_path,
                product_details,
                product_type
            )
            
            if output_path and os.path.exists(output_path):
                log_and_print("INFO", f"Image generated successfully for user {user_id}: {output_path}")
                # Get URLs from image_info
                local_url = image_info.get('local_url')
                public_url = image_info.get('public_url')
                
                # Choose the best URL to share - prefer public URL if available
                share_url = public_url if public_url else local_url
                
                # If share_link_only is True, just send the link instead of the image
                if share_link_only:
                    log_and_print("INFO", f"Sharing image URL instead of image for user {user_id}")
                    
                    # Create a message with both public and local URLs if available
                    url_message = f"🎉 Your marketing image is ready!\n\n"
                    
                    if public_url:
                        url_message += f"📱 Public URL (accessible from any device):\n{public_url}\n\n"
                    
                    if local_url:
                        url_message += f"🔗 Local URL (accessible while our server is running):\n{local_url}\n\n"
                    
                    url_message += f"These links will be available for the next 24 hours.\n\n"
                    url_message += f"To create another image, send 'edit' again."
                    
                    self.whatsapp_client.send_message(
                        from_number,
                        url_message
                    )
                
                return {
                    "success": True,
                    "image_path": output_path,
                    "image_url": share_url,
                    "image_info": image_info
                }
            else:
                log_and_print("ERROR", f"Failed to generate marketing image for user {user_id}")
                
                # Send error message to user
                self.whatsapp_client.send_message(
                    from_number,
                    "❌ Sorry, I couldn't generate the marketing image.\n\n"
                    "This could be due to API limits or issues with the image processing. "
                    "Please try again by sending 'edit'."
                )
                
                return {
                    "success": False,
                    "error": "Failed to generate marketing image"
                }
                
        except Exception as e:
            log_and_print("ERROR", f"Error processing request for user {user_id}: {str(e)}")
            traceback.print_exc()
            
            # Send error message to user
            self.whatsapp_client.send_message(
                from_number,
                "❌ Sorry, there was an unexpected error processing your request.\n\n"
                "Please try again by sending 'edit'."
            )
            
            return {
                "success": False,
                "error": str(e)
            }

    def handle_text_message(self, from_number: str, text: str):
        """Handle incoming text messages with strict user isolation and improved error handling
        
        Args:
            from_number: User's WhatsApp number
            text: Message text
        """
        try:
            print(f"[DEBUG] Received text: '{text}' from {from_number}")
            log_and_print("INFO", f"Processing text message: '{text}' from {from_number}")
            
            # Only process private chats (not groups) and only if from_number is valid
            if not from_number or "@g.us" in from_number:
                log_and_print("INFO", f"Message from group or invalid number: {from_number}. Ignored.")
                return
            
            # Get user's session (creates a new one if needed)
            session = self.session_manager.get_session(from_number)
            user_id = session["user_id"]
            
            log_and_print("INFO", f"Current session state for user {user_id}: {session['state']}")
            
            # Check for cancel command first (this works in any state)
            if text.lower() == 'cancel':
                log_and_print("INFO", f"User {user_id} sent 'cancel' command")
                
                # End current session
                self.session_manager.end_session(from_number)
                
                # Send confirmation
                self.whatsapp_client.send_message(
                    from_number,
                    "✅ Process cancelled.\n\n"
                    "You can start again by sending 'edit' whenever you're ready."
                )
                
                log_and_print("INFO", f"Session cancelled for user {user_id}")
                return
            
            # Check for 'linkmode' command to toggle sharing images as links
            link_commands = ['linkmode', 'share link', 'link mode', 'use link']
            if any(cmd in text.lower() for cmd in link_commands):
                log_and_print("INFO", f"User {user_id} toggled link mode")
                
                # Get current value or default to False
                current_link_mode = session.get('share_link_only', False)
                new_link_mode = not current_link_mode
                
                # Update session with the new value
                self.session_manager.update_session(from_number, {
                    "share_link_only": new_link_mode
                })
                
                # Send confirmation
                if new_link_mode:
                    self.whatsapp_client.send_message(
                        from_number,
                        "✅ Link mode enabled.\n\n"
                        "Your generated images will be shared as links instead of directly sending the images.\n"
                        "This can help prevent cropping or compression by WhatsApp.\n\n"
                        "Send 'linkmode' again to toggle this setting."
                    )
                else:
                    self.whatsapp_client.send_message(
                        from_number,
                        "✅ Link mode disabled.\n\n"
                        "Your generated images will be sent directly to you on WhatsApp.\n\n"
                        "Send 'linkmode' again to toggle this setting."
                    )
                
                log_and_print("INFO", f"Link mode set to {new_link_mode} for user {user_id}")
                return
            
            # Check for edit command - this starts a new session
            if text.lower() == 'edit':
                log_and_print("INFO", f"User {user_id} sent 'edit' command")
                
                # Keep share_link_only setting when starting a new session
                share_link_only = session.get('share_link_only', False)
                
                # Update session to waiting for image
                self.session_manager.update_session(from_number, {
                    "state": "waiting_for_image",
                    "product_image": None,
                    "details": {},
                    "share_link_only": share_link_only  # Preserve this setting
                })
                
                # Send welcome message
                welcome_msg = "Welcome to Marketing Image Editor! 📸\n\n"
                welcome_msg += "Please send your product image to begin.\n\n"
                welcome_msg += "After sending the image, I'll ask for details like company name, product name, price, etc.\n\n"
                
                # Add info about current link mode
                if share_link_only:
                    welcome_msg += "🔗 Link mode is enabled. Your image will be shared as a link.\n\n"
                
                welcome_msg += "You can type 'cancel' at any time to exit the process."
                
                self.whatsapp_client.send_message(
                    from_number,
                    welcome_msg
                )
                
                log_and_print("INFO", f"Sent welcome message to user {user_id}")
                return
            
            # Check for generate command - ONLY VALID in waiting_for_details state
            if text.lower() == 'generate':
                log_and_print("INFO", f"User {user_id} sent 'generate' command")
                
                # Print current session state for debugging
                print(f"[DEBUG] Current session state: {session['state']}")
                print(f"[DEBUG] Product image path: {session.get('product_image')}")
                print(f"[DEBUG] Details collected: {json.dumps(session.get('details', {}))}")
                
                # Validate user is in the right session state
                if session['state'] != 'waiting_for_details':
                    log_and_print("WARNING", f"User {user_id} tried to generate in wrong state: {session['state']}")
                    self.whatsapp_client.send_message(
                        from_number,
                        "Please send 'edit' first to start the image creation process.\n\n"
                        "Then send your product image followed by the required details."
                    )
                    return
                
                # Validate we have all required info
                if not session.get('product_image'):
                    log_and_print("WARNING", f"User {user_id} tried to generate without an image")
                    self.whatsapp_client.send_message(
                        from_number,
                        "Please send a product image first.\n"
                        "To start, send 'edit'.\n\n"
                        "Or type 'cancel' to exit the process."
                    )
                    return
                
                details = session.get('details', {})
                if not all([details.get('company_name'), details.get('product_name'), details.get('price')]):
                    missing = []
                    if not details.get('company_name'):
                        missing.append('company name')
                    if not details.get('product_name'):
                        missing.append('product name')
                    if not details.get('price'):
                        missing.append('price')
                    
                    log_and_print("WARNING", f"User {user_id} tried to generate with missing details: {missing}")
                    self.whatsapp_client.send_message(
                        from_number,
                        f"Missing required details: {', '.join(missing)}\n\n"
                        "Please provide all required information.\n\n"
                        "Type 'cancel' if you want to exit the process."
                    )
                    return
                
                # Process the request
                log_and_print("INFO", f"Starting image generation process for user {user_id}")
                # Get share_link_only setting from session
                share_link_only = session.get('share_link_only', False)
                log_and_print("INFO", f"Share link only mode: {share_link_only}")
                
                result = self.process_request(
                    user_id,
                    from_number,
                    session['product_image'],
                    details,
                    share_link_only=share_link_only
                )
                
                if result['success']:
                    log_and_print("INFO", f"Image generated successfully for user {user_id}: {result['image_path']}")
                    
                    # If in link mode, no need to send image directly - already sent by process_request
                    if share_link_only:
                        log_and_print("INFO", f"Using link-only mode, skipping direct image sending")
                    else:
                        # Get local file path
                        image_path = result['image_path']
                        
                        # Verify the file exists
                        if not os.path.exists(image_path):
                            log_and_print("ERROR", f"Generated image file not found: {image_path}")
                            self.whatsapp_client.send_message(
                                from_number,
                                "Sorry, there was an error saving the generated image. Please try again by sending 'edit'."
                            )
                            return
                        
                        # Read the image for base64 encoding
                        log_and_print("INFO", f"Reading image file for base64 encoding: {image_path}")
                        print(f"[DEBUG] File size: {os.path.getsize(image_path)} bytes")
                        
                        try:
                            # Read image bytes directly without any manipulation
                            with open(image_path, 'rb') as img_file:
                                img_data = img_file.read()
                                print(f"[DEBUG] Image data read: {len(img_data)} bytes")
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                print(f"[DEBUG] Base64 encoding length: {len(img_base64)} characters")
                        except Exception as read_error:
                            log_and_print("ERROR", f"Failed to read image file: {str(read_error)}")
                            traceback.print_exc()
                            self.whatsapp_client.send_message(
                                from_number,
                                "Sorry, I had trouble processing the generated image. Please try again by sending 'edit'."
                            )
                            return
                        
                        # Send the generated image without modifications
                        log_and_print("INFO", f"Sending original unmodified image to user {user_id}")
                        
                        # First check image dimensions
                        try:
                            with Image.open(image_path) as img_check:
                                width, height = img_check.size
                                aspect_ratio = width / height
                                log_and_print("INFO", f"Sending image dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.2f}")
                                
                                # If image might be problematic for WhatsApp, log a warning
                                if width > 2000 or height > 2000:
                                    log_and_print("WARNING", f"Image dimensions ({width}x{height}) might exceed WhatsApp maximum size")
                                if aspect_ratio > 2.5 or aspect_ratio < 0.4:
                                    log_and_print("WARNING", f"Image aspect ratio ({aspect_ratio:.2f}) is extreme and might be cropped by WhatsApp")
                        except Exception as img_check_err:
                            log_and_print("WARNING", f"Could not check image dimensions: {str(img_check_err)}")
                        
                        # Send image with clear MIME type
                        media_result = self.whatsapp_client.send_media(
                            to_number=from_number,
                            caption="🎉 Here's your marketing image!\n\n"
                            "To create another image, send 'edit' again.",
                            media_base64=img_base64,
                            filename=os.path.basename(image_path)
                        )
                        
                        # After sending the direct image, also send links if available
                        image_info = result.get('image_info', {})
                        public_url = image_info.get('public_url')
                        
                        # If we have a public URL, share it for the user's convenience
                        if public_url:
                            self.whatsapp_client.send_message(
                                from_number,
                                f"📱 You can also access your image at this public link:\n{public_url}\n\n"
                                f"This link can be shared with anyone and will work on any device."
                            )
                        
                        # Set media_result for success check below
                        media_result = {"success": True}
                    
                    if media_result.get('success', False) or share_link_only:
                        log_and_print("INFO", f"Image sent successfully to user {user_id}")
                    else:
                        log_and_print("ERROR", f"Failed to send image to user {user_id}: {media_result.get('error')}")
                        # Try to send an error message
                        self.whatsapp_client.send_message(
                            from_number,
                            "I created your marketing image but couldn't send it. Please try again by sending 'edit'."
                        )
                    
                    # End session - IMPORTANT for proper user isolation
                    self.session_manager.end_session(from_number)
                    log_and_print("INFO", f"Session ended for user {user_id} after successful image generation")
                else:
                    log_and_print("ERROR", f"Failed to generate image for user {user_id}: {result.get('error')}")
                    self.whatsapp_client.send_message(
                        from_number,
                        f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                        "Please try again by sending 'edit'."
                    )
                return
            
            # Handle conversation flow based on state
            if session['state'] == 'waiting_for_image':
                log_and_print("INFO", f"User {user_id} sent text while waiting for image")
                self.whatsapp_client.send_message(
                    from_number,
                    "Please send your product image first.\n"
                    "I'm waiting for an image file.\n\n"
                    "Type 'cancel' if you want to exit the process."
                )
                return
                
            elif session['state'] == 'waiting_for_details':
                log_and_print("INFO", f"User {user_id} sent details: {text}")
                # Parse the details
                lines = text.split('\n')
                detail_provided = False
                
                # First, parse details if user sent a structured message
                structured_details = {}
                
                # Check if this is a structured message
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        
                        if 'company' in key:
                            structured_details['company_name'] = value
                            detail_provided = True
                        elif 'product' in key:
                            structured_details['product_name'] = value
                            detail_provided = True
                        elif 'price' in key:
                            structured_details['price'] = value
                            detail_provided = True
                        elif 'tagline' in key:
                            structured_details['tagline'] = value
                            detail_provided = True
                        elif 'address' in key or 'location' in key:
                            structured_details['address'] = value
                            detail_provided = True
                
                # If structured details provided, update session
                if detail_provided:
                    # Get existing details and update with new ones
                    current_details = session.get('details', {})
                    current_details.update(structured_details)
                    
                    # Update session
                    self.session_manager.update_session(from_number, {
                        "details": current_details
                    })
                else:
                    # If no structured details, try to interpret single line input
                    current_details = session.get('details', {})
                    
                    if not current_details.get('company_name'):
                        current_details['company_name'] = text
                        log_and_print("INFO", f"Set company_name: {text}")
                        detail_provided = True
                    elif not current_details.get('product_name'):
                        current_details['product_name'] = text
                        log_and_print("INFO", f"Set product_name: {text}")
                        detail_provided = True
                    elif not current_details.get('price'):
                        current_details['price'] = text
                        log_and_print("INFO", f"Set price: {text}")
                        detail_provided = True
                    elif not current_details.get('tagline'):
                        current_details['tagline'] = text
                        log_and_print("INFO", f"Set tagline: {text}")
                        detail_provided = True
                    elif not current_details.get('address'):
                        current_details['address'] = text
                        log_and_print("INFO", f"Set address: {text}")
                        detail_provided = True
                    
                    # Update session with new details
                    self.session_manager.update_session(from_number, {
                        "details": current_details
                    })
                
                # Send updated status and next step
                log_and_print("INFO", f"Sending status update to user {user_id}")
                
                # Get updated session after changes
                session = self.session_manager.get_session(from_number)
                details = session.get('details', {})
                
                status_msg = "📝 Current details:\n\n"
                status_msg += f"Company: {details.get('company_name', '❌')}\n"
                status_msg += f"Product: {details.get('product_name', '❌')}\n"
                status_msg += f"Price: {details.get('price', '❌')}\n"
                status_msg += f"Tagline: {details.get('tagline', '➖')}\n"
                status_msg += f"Address: {details.get('address', '➖')}\n\n"
                
                # Check what's still needed
                if not details.get('company_name'):
                    status_msg += "👉 Please send your company name.\n"
                elif not details.get('product_name'):
                    status_msg += "👉 Please send your product name.\n"
                elif not details.get('price'):
                    status_msg += "👉 Please send the price.\n"
                else:
                    status_msg += "✅ All required information received!\n\n"
                    status_msg += "To generate the marketing image, send 'generate'\n"
                    status_msg += "To add optional details (tagline, address), just send them."
                
                status_msg += "\n\nYou can type 'cancel' at any time to exit the process."
                
                self.whatsapp_client.send_message(from_number, status_msg)
                return
            
        except Exception as e:
            log_and_print("ERROR", f"Error handling text message from {from_number}: {str(e)}")
            traceback.print_exc()
            try:
                # Get user ID for logging
                user_id = hashlib.md5(from_number.encode()).hexdigest()
                log_and_print("INFO", f"Sending error message to user {user_id}")
                
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, an error occurred. Please try again.\n"
                    "Send 'edit' to start over."
                )
            except Exception as send_error:
                log_and_print("ERROR", f"Failed to send error message: {str(send_error)}")
    
    def handle_image_message(self, from_number: str, webhook_data: Dict):
        """Handle incoming image messages with improved error handling and user isolation
        
        Args:
            from_number: User's WhatsApp number
            webhook_data: Complete webhook data
        """
        try:
            log_and_print("INFO", f"Image received from {from_number}. Processing...")
            
            # Only process private chats (not groups) and only if from_number is valid
            if not from_number or "@g.us" in from_number:
                log_and_print("INFO", f"Message from group or invalid number: {from_number}. Ignored.")
                return
            
            # Get user's session
            session = self.session_manager.get_session(from_number)
            user_id = session["user_id"]
            
            # Debug session state
            log_and_print("INFO", f"Current session state for user {user_id}: {session['state']}")
            
            # Check if we're in the right state to receive an image
            if session['state'] != 'waiting_for_image':
                log_and_print("WARNING", f"User {user_id} sent image but session state is {session['state']}, not waiting_for_image")
                self.whatsapp_client.send_message(
                    from_number,
                    "I wasn't expecting an image right now.\n"
                    "To start the process, please send 'edit' first.\n\n"
                    "You can also type 'cancel' to exit the current process."
                )
                return
            
            # Process the image from webhook data with proper error handling
            try:
                # Extract image data from the webhook data
                log_and_print("INFO", f"Extracting image data from webhook for user {user_id}")
                image_bytes = self.image_handler.extract_image_from_maytapi(webhook_data, from_number)
                
                # Validate the extracted image data
                image_valid = False
                if image_bytes and len(image_bytes) > 100:
                    try:
                        # Verify it's a valid image
                        img = Image.open(BytesIO(image_bytes))
                        img_format = img.format
                        img_size = img.size
                        log_and_print("INFO", f"Successfully extracted valid image: {img_format}, {img_size}px, {len(image_bytes)} bytes")
                        image_valid = True
                    except Exception as img_error:
                        log_and_print("ERROR", f"Extracted data is not a valid image: {str(img_error)}")
                        image_valid = False
                else:
                    log_and_print("ERROR", f"Extracted image data too small or empty: {len(image_bytes) if image_bytes else 0} bytes")
                    image_valid = False
                
                # If we couldn't get valid image data, inform the user
                if not image_valid:
                    log_and_print("WARNING", f"Could not extract valid image data from user {user_id}, sending error message")
                    self.whatsapp_client.send_message(
                        from_number,
                        "⚠️ I received your image but couldn't process it properly.\n\n"
                        "Please try sending the image again, or send a different image.\n\n"
                        "Type 'cancel' if you'd like to exit the process and try again later."
                    )
                    return
                
                # Generate a unique filename and save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"user_{user_id}_{timestamp}.jpg"
                image_path, success = self.image_handler.save_image_from_bytes(image_bytes, user_id, filename)
                
                if not success or not image_path:
                    log_and_print("ERROR", f"Failed to save image for user {user_id}")
                    self.whatsapp_client.send_message(
                        from_number,
                        "Sorry, I couldn't process your image. Please try again by sending a different image.\n\n"
                        "You can type 'cancel' to exit the process."
                    )
                    return
                
                # Update session with image path and move to details state
                self.session_manager.update_session(from_number, {
                    "product_image": image_path,
                    "state": "waiting_for_details"
                })
                
                # Send confirmation and request details
                self.whatsapp_client.send_message(
                    from_number,
                    "✅ Product image received!\n\n"
                    "Now please provide the following details:\n\n"
                    "1️⃣ Company name\n"
                    "2️⃣ Product name\n"
                    "3️⃣ Price\n"
                    "4️⃣ Tagline (optional)\n"
                    "5️⃣ Address (optional)\n\n"
                    "You can send them one by one or all at once.\n"
                    "Example format:\n"
                    "Company: ABC Corp\n"
                    "Product: Premium Coffee\n"
                    "Price: $20\n\n"
                    "When you're ready to generate the image, send 'generate'\n\n"
                    "You can type 'cancel' at any time to exit the process."
                )
                
                log_and_print("INFO", f"Successfully processed image for user {user_id}")
                
            except Exception as process_error:
                log_and_print("ERROR", f"Error processing image for user {user_id}: {str(process_error)}")
                traceback.print_exc()
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your image. Please try again.\n"
                    "Send 'edit' to start over or 'cancel' to exit."
                )
                
        except Exception as e:
            log_and_print("ERROR", f"Error handling image message from {from_number}: {str(e)}")
            traceback.print_exc()
            try:
                self.whatsapp_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your image. Please try again.\n"
                    "Start over by sending 'edit' or type 'cancel' to exit."
                )
            except Exception as send_error:
                log_and_print("ERROR", f"Failed to send error message: {str(send_error)}")
    
    def handle_audio_message(self, from_number: str):
        """Handle incoming audio messages with user isolation
        
        Args:
            from_number: User's WhatsApp number
        """
        try:
            log_and_print("INFO", f"Audio message received from {from_number}")
            
            # Get user's session
            session = self.session_manager.get_session(from_number)
            user_id = session["user_id"]
            
            # Check if user is in an active flow
            if session["state"] != "waiting_for_command":
                self.whatsapp_client.send_message(
                    from_number,
                    "I received your voice message, but I need text input for this process.\n\n"
                    "You can type 'cancel' to exit the current process if needed."
                )
            else:
                # Only send a message if the user has previously sent 'edit'
                if session.get('welcomed', False):
                    # Just send a helpful reminder
                    self.whatsapp_client.send_message(
                        from_number,
                        "I received your voice message, but I work better with text.\n\n"
                        "To create a marketing image, please send 'edit'."
                    )
            
            log_and_print("INFO", f"Audio message handled for user {user_id}")
            
        except Exception as e:
            log_and_print("ERROR", f"Error handling audio message from {from_number}: {str(e)}")
            traceback.print_exc()

###################
# SETUP AND INITIALIZATION
###################

# Initialize marketing bot
log_and_print("INFO", "Initializing marketing bot with environment variables")
marketing_bot = MarketingBot(
    openai_key=OPENAI_API_KEY,
    maytapi_product_id=MAYTAPI_PRODUCT_ID,
    maytapi_api_token=MAYTAPI_API_TOKEN,
    maytapi_phone_id=MAYTAPI_PHONE_ID
)

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    log_and_print("INFO", "Home page accessed")
    return jsonify({
        "service": "Marketing Image Generator (WhatsApp Version)",
        "status": "running",
        "version": "2.0",
        "endpoint": "/webhook"
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    log_and_print("DEBUG", "Health check accessed")
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "openai": OPENAI_API_KEY is not None,
        "maytapi": all([MAYTAPI_API_TOKEN, MAYTAPI_PRODUCT_ID, MAYTAPI_PHONE_ID]),
        "user_sessions": len(marketing_bot.session_manager.sessions),
        "welcomed_users": len(marketing_bot.session_manager.welcomed_users)
    })

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    log_and_print("DEBUG", f"Serving image: {path}")
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via Maytapi webhook"""
    try:
        # Extract the webhook data
        webhook_data = request.json
        log_and_print("INFO", f"Received webhook data of type: {type(webhook_data)}")
        print(f"[DEBUG] Webhook data keys: {list(webhook_data.keys()) if isinstance(webhook_data, dict) else 'Not a dict'}")
        
        # Check if we have a valid message structure
        if not webhook_data or not isinstance(webhook_data, dict):
            log_and_print("ERROR", "Invalid webhook data format")
            return jsonify({"status": "error", "message": "Invalid webhook data format"})
        
        # Extract message type and phone number
        message_type = webhook_data.get('type', '')
        if message_type != 'message':
            log_and_print("INFO", f"Ignoring non-message event: {message_type}")
            return jsonify({"status": "success", "message": "Non-message event ignored"})
        
        # Extract user and message data
        user_data = webhook_data.get('user', {})
        message_data = webhook_data.get('message', {})
        
        # Extract sender info
        from_number = user_data.get('id', '')  # Format: 905301234567@c.us
        phone_number = user_data.get('phone', '')  # Format: 905301234567
        
        # Use phone number if id is not available
        if not from_number and phone_number:
            from_number = f"{phone_number}@c.us"
        
        if not from_number:
            log_and_print("ERROR", "No sender phone number in webhook data")
            return jsonify({"status": "error", "message": "No sender phone number"})
        
        # Check if message is from the bot itself
        is_from_me = message_data.get('fromMe', False)
        if is_from_me:
            log_and_print("INFO", "Ignoring message from the bot itself")
            return jsonify({"status": "success", "message": "Bot message ignored"})
        
        # Extract message ID to prevent duplicate processing
        message_id = message_data.get('id', '')
        
        # Skip if we've already processed this message
        if message_id and message_id in processed_messages:
            log_and_print("INFO", f"Skipping duplicate message: {message_id}")
            return jsonify({"status": "success", "message": "Duplicate message skipped"})
        
        # Mark as processed if it has an ID
        if message_id:
            processed_messages[message_id] = datetime.now().timestamp()
            
            # Limit cache size by removing old entries (keep last 100)
            if len(processed_messages) > 100:
                oldest = sorted(processed_messages.items(), key=lambda x: x[1])[0][0]
                processed_messages.pop(oldest)
        
        # Extract message content
        content_type = message_data.get('type', '')
        log_and_print("INFO", f"Message content type: {content_type}")
        
        # Handle media messages (images)
        if content_type == 'image':
            log_and_print("INFO", f"Detected image message from {from_number}")
            # Log additional message data for debugging
            log_and_print("DEBUG", f"Message data: {json.dumps(message_data, indent=2)}")
            
            # Check for direct media URL
            if 'media' in message_data:
                log_and_print("INFO", f"Found media URL in message: {message_data['media'][:50]}...")
            
            # Pass full webhook data to handle_image_message
            marketing_bot.handle_image_message(from_number, webhook_data)
            return jsonify({"status": "success", "message": "Image processed"})
        
        # Handle audio messages
        elif content_type in ['audio', 'voice', 'ptt']:
            log_and_print("INFO", f"Detected audio message from {from_number}")
            marketing_bot.handle_audio_message(from_number)
            return jsonify({"status": "success", "message": "Audio processed"})
        
        # Handle document messages that might be images
        elif content_type == 'document':
            log_and_print("INFO", f"Detected document message from {from_number}")
            # Check if it's an image document
            mime_type = message_data.get('mimetype', '')
            if mime_type and mime_type.startswith('image/'):
                log_and_print("INFO", f"Document is an image, processing as image")
                marketing_bot.handle_image_message(from_number, webhook_data)
            else:
                log_and_print("INFO", f"Document is not an image, sending reminder")
                marketing_bot.handle_audio_message(from_number)  # Reuse audio handler for generic response
            return jsonify({"status": "success", "message": "Document processed"})
        
        # Handle text messages
        elif content_type == 'text':
            body = message_data.get('text', '')
            log_and_print("INFO", f"Detected text message from {from_number}: {body}")
            marketing_bot.handle_text_message(from_number, body)
            return jsonify({"status": "success", "message": "Text processed"})
        
        # Fallback for other message types
        else:
            log_and_print("WARNING", f"Unhandled message type: {content_type}")
            # Send a generic response for unhandled types
            marketing_bot.handle_audio_message(from_number)  # Reuse audio handler for generic response
            return jsonify({"status": "success", "message": f"Unhandled message type: {content_type}"})
    
    except Exception as e:
        log_and_print("ERROR", f"Webhook error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

###################
# MAINTENANCE TASKS
###################

@app.route('/maintenance/cleanup', methods=['POST'])
def cleanup_sessions():
    """Cleanup old sessions and processed messages"""
    try:
        # Get cleanup hours from request or use default
        hours = request.json.get('hours', 24) if request.is_json else 24
        
        # Clean up old sessions
        session_count = marketing_bot.session_manager.clean_old_sessions(hours)
        
        # Clean up old processed messages
        current_time = datetime.now().timestamp()
        message_ids = list(processed_messages.keys())
        message_count = 0
        
        for msg_id in message_ids:
            if current_time - processed_messages[msg_id] > hours * 3600:
                processed_messages.pop(msg_id)
                message_count += 1
        
        log_and_print("INFO", f"Cleaned up {session_count} sessions and {message_count} processed messages older than {hours} hours")
        
        return jsonify({
            "status": "success",
            "cleaned_sessions": session_count,
            "cleaned_messages": message_count,
            "hours": hours
        })
    
    except Exception as e:
        log_and_print("ERROR", f"Maintenance cleanup error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})

@app.route('/maintenance/reset', methods=['POST'])
def reset_welcomed_users():
    """Reset the welcomed users registry for testing"""
    try:
        before_count = len(marketing_bot.session_manager.welcomed_users)
        marketing_bot.session_manager.welcomed_users.clear()
        log_and_print("INFO", f"Reset welcomed users registry. Before: {before_count}, After: {len(marketing_bot.session_manager.welcomed_users)}")
        
        return jsonify({
            "status": "success",
            "before_count": before_count,
            "after_count": len(marketing_bot.session_manager.welcomed_users)
        })
    except Exception as e:
        log_and_print("ERROR", f"Reset welcomed users error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/maintenance/sessions', methods=['GET'])
def view_sessions():
    """View active sessions (for debugging only)"""
    try:
        sessions = marketing_bot.session_manager.get_all_sessions()
        sanitized_sessions = {}
        
        # Remove sensitive data for display
        for user_id, session in sessions.items():
            sanitized_session = {
                "user_id": user_id,
                "state": session.get("state"),
                "has_image": bool(session.get("product_image")),
                "details_count": len(session.get("details", {})),
                "welcomed": session.get("welcomed", False),
                "last_active": session.get("last_active")
            }
            sanitized_sessions[user_id] = sanitized_session
        
        return jsonify({
            "status": "success",
            "active_sessions": len(sanitized_sessions),
            "welcomed_users": len(marketing_bot.session_manager.welcomed_users),
            "sessions": sanitized_sessions
        })
    except Exception as e:
        log_and_print("ERROR", f"View sessions error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Print environment info for debugging
    print("-" * 50)
    print("MARKETING BOT STARTUP")
    print("-" * 50)
    print(f"Python version: {os.sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Environment variables: PORT={port}")
    print(f"OpenAI API Key Present: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"Maytapi API Token Present: {'Yes' if MAYTAPI_API_TOKEN else 'No'}")
    print(f"Maytapi Product ID: {MAYTAPI_PRODUCT_ID}")
    print(f"Maytapi Phone ID: {MAYTAPI_PHONE_ID}")
    print(f"Checking directories...")
    
    # Make sure directories exist and are writable
    for directory in ['images', 'images/input', 'images/output']:
        os.makedirs(directory, exist_ok=True)
        print(f"Directory '{directory}' exists: {os.path.exists(directory)}")
    
    # Test PIL import
    print(f"PIL/Pillow version: {Image.__version__ if hasattr(Image, '__version__') else 'unknown'}")
    
    # Log startup
    log_and_print("INFO", f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    print("-" * 50)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production

def upload_image_to_imgur(image_bytes: bytes) -> Optional[str]:
    """Upload an image to Imgur and return the shareable link
    
    Args:
        image_bytes: The image data as bytes
        
    Returns:
        Optional[str]: The URL to the uploaded image, or None if upload failed
    """
    if not IMGUR_CLIENT_ID:
        log_and_print("WARNING", "Imgur upload requested but IMGUR_CLIENT_ID not configured")
        return None
    
    try:
        log_and_print("INFO", "Uploading image to Imgur...")
        headers = {
            'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'
        }
        
        # Convert image bytes to base64
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        data = {
            'image': b64_image,
            'type': 'base64',
            'title': f'Marketing Image {datetime.now().strftime("%Y-%m-%d")}',
            'description': 'Generated by Marketing Bot'
        }
        
        response = requests.post(
            'https://api.imgur.com/3/image',
            headers=headers,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                imgur_url = result.get('data', {}).get('link')
                log_and_print("INFO", f"Image uploaded to Imgur: {imgur_url}")
                return imgur_url
        
        log_and_print("ERROR", f"Failed to upload to Imgur: HTTP {response.status_code}")
        if hasattr(response, 'text'):
            log_and_print("ERROR", f"Imgur response: {response.text[:200]}...")
        return None
    
    except Exception as e:
        log_and_print("ERROR", f"Error uploading to Imgur: {str(e)}")
        return None