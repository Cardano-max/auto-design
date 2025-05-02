#!/usr/bin/env python3
"""
Marketing Image Generator - WaAPI WhatsApp Version
--------------------------------------------
Production version with WaAPI WhatsApp API integration
Configured for Railway.com deployment
Updated for GPT-Image-1 model implementation
"""

import os
import base64
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Optional, Dict, List, Union
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to catch all logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")

# Set up debug printing function with timestamp
def debug_print(message):
    """Print debug message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[DEBUG][{timestamp}] {message}")

# Utility function to log and print messages
def log_and_print(level, message):
    """Log and print messages with standardized format"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[{level}][{timestamp}] {message}")
    
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)
os.makedirs('images/masks', exist_ok=True)
os.makedirs('images/logos', exist_ok=True)

# Store user sessions
user_sessions = {}

# Store processed messages to prevent duplicates
processed_messages = {}

# Store last message time for rate limiting
last_message_time = {}

# Initialize OpenAI with API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

# Initialize OpenAI client
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)
logger.info("Using OpenAI client version")
debug_print(f"Initialized OpenAI client with API key: {OPENAI_API_KEY[:5]}...")
debug_print(f"OpenAI client version: {OpenAI.__version__ if hasattr(OpenAI, '__version__') else 'unknown'}")

# WaAPI configuration - from environment variables
WAAPI_API_TOKEN = os.getenv('WAAPI_API_TOKEN')
WAAPI_INSTANCE_ID = os.getenv('WAAPI_INSTANCE_ID')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')

# Validate required environment variables
required_vars = {
    'WAAPI_API_TOKEN': WAAPI_API_TOKEN,
    'WAAPI_INSTANCE_ID': WAAPI_INSTANCE_ID
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info(f"Environment variables loaded. Instance ID: {WAAPI_INSTANCE_ID}")

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with product details"""
        logger.debug(f"Generating master template with product details: {product_details}")
        return f"""Create a professional marketing poster for this product.

PRESERVE THE PRODUCT: Keep the product image as the central focus, but remove its background and place it on a new background.

ADD THESE MARKETING ELEMENTS:
- Company name: "{product_details.get('company_name', '')}" - place at top of poster
- Product name: "{product_details.get('product_name', '')}" - place below the product 
- Price: "{product_details.get('price', '')}" - place in an eye-catching circle in bottom right
- Tagline: "{product_details.get('tagline', '')}" - place between company name and product
- Address: "{product_details.get('address', '')}" - place at bottom in smaller text

DESIGN STYLE:
- Use a clean, modern marketing layout 
- Add a subtle gradient background that complements the product colors
- Use professional typography with clear hierarchy
- Ensure all text is perfectly readable
- Make it look like a high-end advertising poster created by a professional graphic designer

The final image should look like it was designed by a professional marketing agency.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template"""
        logger.debug(f"Generating beverage template with product details: {product_details}")
        return f"""Create a premium café/beverage marketing poster.

PRESERVE THE BEVERAGE: Keep the drink as the central focus, removing its background. Enhance the appearance to make it look appetizing.

ADD THESE MARKETING ELEMENTS:
- Company name: "{product_details.get('company_name', '')}" - place at top in elegant script font
- Product name: "{product_details.get('product_name', '')}" - place below the beverage
- Price: "{product_details.get('price', '')}" - place in an eye-catching circle in bottom right
- Tagline: "{product_details.get('tagline', '')}" - place between company name and product
- Address: "{product_details.get('address', '')}" - place at bottom in smaller text

DESIGN STYLE:
- Warm, inviting color palette that complements the drink
- Add subtle background elements like coffee beans or ingredients
- Use professional café marketing aesthetics similar to premium coffee shops
- Create a gentle vignette effect to focus attention on the drink
- Ensure all text is perfectly readable with good contrast

The final image should look like a professional café marketing poster.
"""

    @staticmethod
    def get_food_template(product_details: Dict) -> str:
        """Generate food-specific prompt template"""
        logger.debug(f"Generating food template with product details: {product_details}")
        return f"""Create a mouth-watering professional food marketing poster.

PRESERVE THE FOOD ITEM: Keep the food as the central focus, removing its background. Enhance its appearance to make it look delicious.

ADD THESE MARKETING ELEMENTS:
- Company name: "{product_details.get('company_name', '')}" - place at top in clean font
- Product name: "{product_details.get('product_name', '')}" - place below the food in bold font
- Price: "{product_details.get('price', '')}" - place in an eye-catching badge
- Tagline: "{product_details.get('tagline', '')}" - place between company name and food
- Address: "{product_details.get('address', '')}" - place at bottom in smaller text

DESIGN STYLE:
- Use colors that complement the food and stimulate appetite
- Add subtle food-related elements in the background
- Professional food photography style with perfect presentation
- Create a balanced composition with the food as the hero
- Ensure all text is clear and readable

The final image should look like a professional restaurant or food marketing poster.
"""

    @staticmethod
    def get_mask_generation_prompt(product_type: str) -> str:
        """Generate a prompt for creating a mask for the product"""
        logger.debug(f"Generating mask prompt for product type: {product_type}")
        
        base_prompt = "Generate a precise mask image to isolate the main product in the picture. "
        
        if product_type.lower() == "beverage":
            specific_prompt = "Create a mask where the beverage container and all its contents are WHITE and everything else is BLACK. Include all parts of the beverage - the cup/glass, liquid, toppings, straws, and garnishes."
        elif product_type.lower() == "food":
            specific_prompt = "Create a mask where the food item and all its components are WHITE and everything else is BLACK. Include the plate/container if it's part of the presentation."
        else:
            specific_prompt = "Create a mask where the main product is WHITE and everything else is BLACK. Be precise with the product edges."
        
        return f"{base_prompt}{specific_prompt} This mask will be used to preserve only the product while changing the background. The mask should be the same size as the input image."

###################
# IMAGE GENERATION
###################

class ImageGenerator:
    def __init__(self, api_key: str):
        log_and_print("INFO", f"Initializing ImageGenerator with API key: {api_key[:5]}...")
        self.api_key = api_key
        self.client = openai_client
        log_and_print("INFO", "ImageGenerator initialized with OpenAI API - gpt-image-1 model")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 string"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    
    def generate_mask(self, product_image_path: str, product_type: str = "generic") -> Optional[str]:
        """Generate a mask for the product image"""
        try:
            log_and_print("INFO", f"Generating mask for image at {product_image_path}")
            
            # Verify image exists
            if not os.path.exists(product_image_path):
                log_and_print("ERROR", f"Product image not found at path: {product_image_path}")
                return None
            
            # Get mask generation prompt
            prompt = PromptTemplates.get_mask_generation_prompt(product_type)
            log_and_print("INFO", f"Using mask prompt: {prompt}")
            
            # Prepare the image for the API
            with open(product_image_path, "rb") as img_file:
                # Generate the mask using GPT-Image-1
                max_retries = 3
                retry_delay = 2
                
                for retry in range(max_retries):
                    try:
                        log_and_print("INFO", f"Mask generation API call attempt {retry+1}")
                        
                        # Make the API call for mask generation
                        result = self.client.images.edit(
                            model="gpt-image-1",
                            image=img_file,
                            prompt=prompt,
                            size="1024x1024",
                            response_format="b64_json"
                        )
                        
                        log_and_print("INFO", "Mask generation API call successful")
                        break
                    except Exception as api_error:
                        log_and_print("WARNING", f"Mask generation API call attempt {retry+1} failed: {str(api_error)}")
                        
                        if retry < max_retries - 1:
                            log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        else:
                            log_and_print("ERROR", "All mask generation API call retries failed")
                            return None
            
            # Process the result
            if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                log_and_print("INFO", "Mask image data received")
                
                # Decode the base64 mask image
                b64_data = result.data[0].b64_json
                image_bytes = base64.b64decode(b64_data)
                
                # Save the mask
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                mask_filename = f"mask_{os.path.basename(product_image_path).split('.')[0]}_{timestamp}.png"
                mask_path = os.path.join("images/masks", mask_filename)
                
                # Ensure masks directory exists
                os.makedirs("images/masks", exist_ok=True)
                
                # Save mask image
                with open(mask_path, "wb") as f:
                    f.write(image_bytes)
                
                log_and_print("INFO", f"Mask saved to {mask_path}")
                
                # Convert the mask to have an alpha channel
                try:
                    # Load as grayscale image
                    mask = Image.open(mask_path).convert("L")
                    
                    # Convert to RGBA
                    mask_rgba = mask.convert("RGBA")
                    
                    # Use the mask itself to fill alpha channel
                    mask_rgba.putalpha(mask)
                    
                    # Save with alpha channel
                    alpha_mask_path = f"{os.path.splitext(mask_path)[0]}_alpha.png"
                    mask_rgba.save(alpha_mask_path, format="PNG")
                    
                    log_and_print("INFO", f"Mask with alpha channel saved to {alpha_mask_path}")
                    return alpha_mask_path
                    
                except Exception as mask_error:
                    log_and_print("ERROR", f"Failed to convert mask to alpha channel: {str(mask_error)}")
                    return mask_path
            else:
                log_and_print("ERROR", "No mask image data in response")
                return None
                
        except Exception as e:
            log_and_print("ERROR", f"Error generating mask: {str(e)}")
            traceback.print_exc()
            return None
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, 
                              product_type: str = "beverage", logo_image_path: Optional[str] = None) -> Optional[str]:
        """Generate a marketing image using OpenAI API with gpt-image-1 model"""
        try:
            # Start timing for performance tracking
            start_time = time.time()
            product_name = product_details.get('product_name', 'product')
            
            log_and_print("INFO", f"Starting image generation for {product_name} as {product_type}")
            log_and_print("INFO", f"Input image path: {product_image_path}")
            log_and_print("INFO", f"Logo image path: {logo_image_path}")
            log_and_print("INFO", f"Product details: {json.dumps(product_details)}")
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
                log_and_print("INFO", "Using beverage template")
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
                log_and_print("INFO", "Using food template")
            else:
                prompt = PromptTemplates.get_master_template(product_details)
                log_and_print("INFO", "Using master template")
            
            log_and_print("INFO", f"Using prompt: {prompt}")
            
            # Check if we should use multi-image approach (with logo) or mask approach
            if logo_image_path and os.path.exists(logo_image_path):
                log_and_print("INFO", "Using multi-image approach with logo")
                return self._generate_with_logo(product_image_path, logo_image_path, prompt, product_details)
            else:
                log_and_print("INFO", "Using mask approach (no logo provided)")
                return self._generate_with_mask(product_image_path, prompt, product_details, product_type)
                
        except Exception as e:
            log_and_print("ERROR", f"Error generating marketing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_with_logo(self, product_image_path: str, logo_image_path: str, 
                         prompt: str, product_details: Dict) -> Optional[str]:
        """Generate marketing image with product and logo images"""
        try:
            log_and_print("INFO", "Starting image generation with product and logo")
            
            # Verify images exist
            if not os.path.exists(product_image_path):
                log_and_print("ERROR", f"Product image not found: {product_image_path}")
                return None
            
            if not os.path.exists(logo_image_path):
                log_and_print("ERROR", f"Logo image not found: {logo_image_path}")
                return None
            
            # Add instruction to incorporate the logo
            enhanced_prompt = f"{prompt}\n\nADDITIONAL INSTRUCTIONS: Incorporate the provided logo image appropriately into the design. Place it in a visible but not overwhelming position, likely near the company name or in a corner of the poster."
            
            # Generate with multiple images
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    log_and_print("INFO", f"Multi-image API call attempt {retry+1}")
                    
                    # Open both image files
                    with open(product_image_path, "rb") as product_file, open(logo_image_path, "rb") as logo_file:
                        # Pass multiple images to the API
                        result = self.client.images.edit(
                            model="gpt-image-1",
                            image=[product_file, logo_file],
                            prompt=enhanced_prompt,
                            size="1024x1024",
                            response_format="b64_json"
                        )
                    
                    log_and_print("INFO", "Multi-image API call successful")
                    break
                except Exception as api_error:
                    log_and_print("WARNING", f"Multi-image API call attempt {retry+1} failed: {str(api_error)}")
                    
                    if retry < max_retries - 1:
                        log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        log_and_print("ERROR", "All multi-image API call retries failed")
                        # Fall back to mask approach
                        log_and_print("INFO", "Falling back to mask approach")
                        return self._generate_with_mask(product_image_path, prompt, product_details, "generic")
            
            # Process and save the result
            return self._process_and_save_result(result, product_details)
            
        except Exception as e:
            log_and_print("ERROR", f"Error in multi-image generation: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_with_mask(self, product_image_path: str, prompt: str, 
                         product_details: Dict, product_type: str) -> Optional[str]:
        """Generate marketing image using mask approach"""
        try:
            log_and_print("INFO", "Starting image generation with mask approach")
            
            # Generate a mask for the product
            mask_path = self.generate_mask(product_image_path, product_type)
            if not mask_path:
                log_and_print("ERROR", "Failed to generate mask, attempting without mask")
                
                # Try simple edit without mask as fallback
                return self._generate_simple_edit(product_image_path, prompt, product_details)
            
            log_and_print("INFO", f"Using mask at {mask_path}")
            
            # Generate with mask
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    log_and_print("INFO", f"Masked image API call attempt {retry+1}")
                    
                    # Open image and mask files
                    with open(product_image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                        # Use masked editing
                        result = self.client.images.edit(
                            model="gpt-image-1",
                            image=img_file,
                            mask=mask_file,
                            prompt=prompt,
                            size="1024x1024",
                            response_format="b64_json"
                        )
                    
                    log_and_print("INFO", "Masked image API call successful")
                    break
                except Exception as api_error:
                    log_and_print("WARNING", f"Masked image API call attempt {retry+1} failed: {str(api_error)}")
                    
                    if retry < max_retries - 1:
                        log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        log_and_print("ERROR", "All masked image API call retries failed")
                        # Fall back to simple edit
                        log_and_print("INFO", "Falling back to simple edit approach")
                        return self._generate_simple_edit(product_image_path, prompt, product_details)
            
            # Process and save the result
            return self._process_and_save_result(result, product_details)
            
        except Exception as e:
            log_and_print("ERROR", f"Error in masked image generation: {str(e)}")
            traceback.print_exc()
            return None
    
    def _generate_simple_edit(self, product_image_path: str, prompt: str, 
                           product_details: Dict) -> Optional[str]:
        """Generate marketing image using simple edit approach (fallback)"""
        try:
            log_and_print("INFO", "Starting image generation with simple edit approach")
            
            # Use simple edit as last resort
            max_retries = 3
            retry_delay = 2
            
            for retry in range(max_retries):
                try:
                    log_and_print("INFO", f"Simple edit API call attempt {retry+1}")
                    
                    # Open image file
                    with open(product_image_path, "rb") as img_file:
                        # Use simple editing
                        result = self.client.images.edit(
                            model="gpt-image-1",
                            image=img_file,
                            prompt=prompt,
                            size="1024x1024",
                            response_format="b64_json"
                        )
                    
                    log_and_print("INFO", "Simple edit API call successful")
                    break
                except Exception as api_error:
                    log_and_print("WARNING", f"Simple edit API call attempt {retry+1} failed: {str(api_error)}")
                    
                    if retry < max_retries - 1:
                        log_and_print("INFO", f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        log_and_print("ERROR", "All simple edit API call retries failed")
                        return None
            
            # Process and save the result
            return self._process_and_save_result(result, product_details)
            
        except Exception as e:
            log_and_print("ERROR", f"Error in simple edit generation: {str(e)}")
            traceback.print_exc()
            return None
    
    def _process_and_save_result(self, result, product_details: Dict) -> Optional[str]:
        """Process API result and save the generated image"""
        try:
            # Process the result
            if hasattr(result, 'data') and len(result.data) > 0 and hasattr(result.data[0], 'b64_json'):
                log_and_print("INFO", "Image data received, decoding")
                
                # Decode the base64 image
                b64_data = result.data[0].b64_json
                image_bytes = base64.b64decode(b64_data)
                
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = ''.join(c if c.isalnum() else '_' for c in product_details.get('product_name', 'product'))[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                # Ensure output directory exists
                os.makedirs("images/output", exist_ok=True)
                
                # Save image
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                
                log_and_print("INFO", f"Marketing image saved to {output_path}")
                return output_path
            else:
                log_and_print("ERROR", "No image data in response")
                return None
                
        except Exception as e:
            log_and_print("ERROR", f"Error processing and saving result: {str(e)}")
            traceback.print_exc()
            return None

###################
# WAAPI WHATSAPP API
###################

class WaAPIClient:
    def __init__(self, api_token: str, instance_id: str):
        logger.info(f"Initializing WaAPIClient for instance {instance_id}")
        self.api_token = api_token
        self.instance_id = instance_id
        self.api_base_url = "https://waapi.app/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        logger.info("WaAPIClient initialized successfully")
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make a request to the WaAPI API"""
        url = f"{self.api_base_url}/{endpoint}"
        logger.info(f"Making {method} request to {endpoint}")
        try:
            if method.lower() == "get":
                response = requests.get(url, headers=self.headers)
            elif method.lower() == "post":
                logger.debug(f"POST data: {json.dumps(data)}")
                response = requests.post(url, headers=self.headers, json=data)
            elif method.lower() == "put":
                response = requests.put(url, headers=self.headers, json=data)
            elif method.lower() == "delete":
                response = requests.delete(url, headers=self.headers)
            else:
                logger.error(f"Invalid method: {method}")
                return {"success": False, "error": "Invalid method"}
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response: {json.dumps(result)}")
            return result
        except Exception as e:
            logger.error(f"Error making request to WaAPI: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_instance_status(self) -> Dict:
        """Get the status of the instance"""
        logger.info("Checking instance status")
        return self._make_request("get", f"instances/{self.instance_id}/client/status")
    
    def send_message(self, to: str, message: str) -> Dict:
        """Send a text message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        current_time = time.time()
        if to in last_message_time and current_time - last_message_time[to] < 2:
            logger.info(f"Rate limiting message to {to} - too soon after last message")
            return {"success": False, "error": "Rate limited"}
        
        data = {
            "chatId": to,
            "message": message
        }
        
        logger.info(f"Sending message to {to}: {message[:50]}{'...' if len(message) > 50 else ''}")
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-message", 
            data
        )
        
        if result.get("status") == "success":
            logger.info(f"Message sent successfully to {to}")
            # Update rate limiting
            last_message_time[to] = current_time
            return {"success": True, "data": result.get("data", {})}
        else:
            logger.error(f"Error sending message: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def send_media(self, to: str, caption: str = "", media_url: str = None, media_base64: str = None, 
                  filename: str = None, is_sticker: bool = False) -> Dict:
        """Send a media message via WhatsApp"""
        # Format phone number to WhatsApp format if needed
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        current_time = time.time()
        if to in last_message_time and current_time - last_message_time[to] < 2:
            logger.info(f"Rate limiting media message to {to} - too soon after last message")
            return {"success": False, "error": "Rate limited"}
        
        data = {
            "chatId": to,
            "mediaCaption": caption
        }
        
        # Add either URL or base64 data
        if media_url:
            logger.info(f"Sending media from URL to {to}")
            data["mediaUrl"] = media_url
        elif media_base64:
            logger.info(f"Sending media from base64 to {to}")
            data["mediaBase64"] = media_base64
            # Filename is required for base64 media
            data["mediaName"] = filename or f"media_{int(datetime.now().timestamp())}.png"
        else:
            logger.error("No media URL or base64 data provided")
            return {"success": False, "error": "No media provided"}
        
        # Set if image should be sent as sticker
        if is_sticker:
            data["asSticker"] = True
            
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-media", 
            data
        )
        
        if result.get("status") == "success":
            logger.info(f"Media sent successfully to {to}")
            # Update rate limiting
            last_message_time[to] = current_time
            return {"success": True, "data": result.get("data", {})}
        else:
            logger.error(f"Error sending media: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}
    
    def _format_phone_number(self, phone_number: str) -> str:
        """Format phone number to the expected format for WhatsApp"""
        logger.debug(f"Formatting phone number: {phone_number}")
        # Remove any non-numeric characters except the @ if it's already formatted
        if '@c.us' in phone_number:
            return phone_number
            
        # Strip any non-numeric characters (like +, spaces, etc.)
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Format for WhatsApp: number@c.us
        result = f"{clean_number}@c.us"
        logger.debug(f"Formatted phone number: {result}")
        return result
    
    def is_registered_user(self, phone_number: str) -> bool:
        """Check if a phone number is registered on WhatsApp"""
        # Format the phone number
        contact_id = self._format_phone_number(phone_number)
        logger.info(f"Checking if {contact_id} is registered on WhatsApp")
        
        data = {
            "contactId": contact_id
        }
        
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/is-registered-user", 
            data
        )
        
        if result.get("status") == "success" and "data" in result:
            is_registered = result["data"].get("isRegisteredUser", False)
            logger.info(f"User {contact_id} is registered: {is_registered}")
            return is_registered
        
        logger.error(f"Failed to check if user is registered: {result.get('error', 'Unknown error')}")
        return False

###################
# MARKETING BOT
###################

class MarketingBot:
    def __init__(self, openai_key: str, waapi_token: str, waapi_instance_id: str):
        logger.info("Initializing MarketingBot")
        self.image_generator = ImageGenerator(openai_key)
        self.waapi_client = WaAPIClient(waapi_token, waapi_instance_id)
        logger.info("MarketingBot initialized with OpenAI and WaAPI")
    
    def process_request(self, user_id: str, product_image_path: str, product_details: Dict, 
                       product_type: str = "beverage", logo_image_path: Optional[str] = None) -> Dict:
        """Process a marketing image request"""
        try:
            logger.info(f"Processing request for user {user_id}")
            debug_print(f"Processing request for user {user_id}")
            debug_print(f"Product image path: {product_image_path}")
            debug_print(f"Logo image path: {logo_image_path if logo_image_path else 'None'}")
            debug_print(f"Product details: {json.dumps(product_details)}")
            debug_print(f"Product type: {product_type}")
            
            # Check if product image exists
            if not os.path.exists(product_image_path):
                debug_print(f"ERROR: Product image does not exist: {product_image_path}")
                return {"success": False, "error": "Product image file not found"}
            
            # Check logo image if provided
            if logo_image_path and not os.path.exists(logo_image_path):
                debug_print(f"ERROR: Logo image does not exist: {logo_image_path}")
                return {"success": False, "error": "Logo image file not found"}
            
            # Check file size
            file_size = os.path.getsize(product_image_path)
            debug_print(f"Product image file size: {file_size} bytes")
            
            # Try to open the image to verify it's valid
            try:
                with Image.open(product_image_path) as img:
                    debug_print(f"Image opened successfully: {img.format}, {img.size}, {img.mode}")
            except Exception as img_error:
                debug_print(f"Error opening image: {str(img_error)}")
            
            # Generate marketing image
            logger.info("Starting image generation")
            debug_print("Starting image generation with gpt-image-1 model")
            output_path = self.image_generator.generate_marketing_image(
                product_image_path,
                product_details,
                product_type,
                logo_image_path
            )
            
            if output_path:
                logger.info(f"Image generated successfully: {output_path}")
                # Get the Railway app URL for serving images
                app_url = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'https://auto-design-production.up.railway.app'))
                image_url = f"{app_url}/images/output/{os.path.basename(output_path)}"
                logger.info(f"Image URL: {image_url}")
                
                return {
                    "success": True,
                    "image_path": output_path,
                    "image_url": image_url
                }
            else:
                logger.error("Failed to generate marketing image")
                return {
                    "success": False,
                    "error": "Failed to generate marketing image"
                }
                
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Initialize marketing bot
logger.info("Initializing marketing bot with environment variables")
marketing_bot = MarketingBot(
    openai_key=OPENAI_API_KEY,
    waapi_token=WAAPI_API_TOKEN,
    waapi_instance_id=WAAPI_INSTANCE_ID
)

###################
# MESSAGE HANDLERS
###################

def handle_text_message(from_number: str, text: str):
    """Handle incoming text messages"""
    try:
        logger.info(f"Processing text message: '{text}' from {from_number}")
        debug_print(f"Processing text message: '{text}' from {from_number}")
        
        # Create session if not exists
        if from_number not in user_sessions:
            logger.info(f"Creating new session for {from_number}")
            debug_print(f"Creating new session for {from_number}")
            user_sessions[from_number] = {
                "product_image": None,
                "logo_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
        
        session = user_sessions[from_number]
        logger.info(f"Current session state: {session['state']}")
        
        # Check for start command
        if text.lower() == 'edit':
            logger.info(f"User {from_number} sent 'edit' command")
            session['state'] = 'waiting_for_image'
            session['product_image'] = None
            session['logo_image'] = None
            session['details'] = {}
            logger.info(f"Session state changed to 'waiting_for_image'")
            
            marketing_bot.waapi_client.send_message(
                from_number,
                "Welcome to Marketing Image Editor! 📸\n\n"
                "Please send your product image to begin.\n\n"
                "After sending the product image, you can optionally send a logo image, and then I'll ask for details like company name, product name, price, etc."
            )
            logger.info(f"Sent welcome message to {from_number}")
            return
        
        # Check for generate command
        if text.lower() == 'generate':
            logger.info(f"User {from_number} sent 'generate' command")
            debug_print(f"User {from_number} sent 'generate' command")
            debug_print(f"Current session state: {session['state']}")
            debug_print(f"Session details: {json.dumps(session)}")
            
            # Validate we have all required info
            if not session.get('product_image'):
                logger.warning(f"User {from_number} tried to generate without an image")
                debug_print(f"Error: No product image found in session")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "Please send a product image first.\n"
                    "To start, send 'edit'."
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
                
                logger.warning(f"User {from_number} tried to generate with missing details: {missing}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"Missing required details: {', '.join(missing)}\n\n"
                    "Please provide all required information."
                )
                return
            
            # Generate the image
            logger.info(f"Sending generation message to {from_number}")
            marketing_bot.waapi_client.send_message(
                from_number,
                "✨ Generating your marketing image..."
            )
            
            # Process in the background
            logger.info(f"Starting image generation process for {from_number}")
            debug_print(f"Starting image generation process for {from_number}")
            debug_print(f"Product image path: {session['product_image']}")
            debug_print(f"Logo image path: {session.get('logo_image', 'None')}")
            debug_print(f"Product details: {json.dumps(details)}")
            
            try:
                # Add a check to verify the product image exists
                if not os.path.exists(session['product_image']):
                    debug_print(f"WARNING: Product image file does not exist: {session['product_image']}")
                    # Create a dummy file if needed for testing
                    with open(session['product_image'], 'wb') as f:
                        f.write(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="))
                    debug_print(f"Created placeholder image at: {session['product_image']}")
            except Exception as img_check_error:
                debug_print(f"Error checking product image: {str(img_check_error)}")
            
            result = marketing_bot.process_request(
                from_number,
                session['product_image'],
                details,
                product_type="beverage",  # Default to beverage, can be customized
                logo_image_path=session.get('logo_image')
            )
            
            if result['success']:
                logger.info(f"Image generated successfully for {from_number}: {result['image_path']}")
                # Get local file path
                image_path = result['image_path']
                
                # Read the image for base64 encoding
                logger.info(f"Reading image file for base64 encoding: {image_path}")
                with open(image_path, 'rb') as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Send the generated image
                logger.info(f"Sending generated image to {from_number}")
                media_result = marketing_bot.waapi_client.send_media(
                    from_number,
                    "🎉 Here's your marketing image!\n\n"
                    "To create another image, send 'edit' again.",
                    media_base64=img_base64,
                    filename=os.path.basename(image_path)
                )
                
                if media_result['success']:
                    logger.info(f"Image sent successfully to {from_number}")
                else:
                    logger.error(f"Failed to send image: {media_result.get('error')}")
                    # Try to send an error message
                    marketing_bot.waapi_client.send_message(
                        from_number,
                        "I created your marketing image but couldn't send it. Please try again by sending 'edit'."
                    )
                
                # Reset state
                session['state'] = 'waiting_for_command'
                logger.info(f"Session state reset to 'waiting_for_command'")
            else:
                logger.error(f"Failed to generate image: {result.get('error')}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"Sorry, I couldn't generate the image: {result.get('error', 'Unknown error')}\n\n"
                    "Please try again or send 'edit' to start over."
                )
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            logger.info(f"User {from_number} sent text while waiting for image")
            marketing_bot.waapi_client.send_message(
                from_number,
                "Please send your product image first.\n"
                "I'm waiting for an image file."
            )
            return
        
        elif session['state'] == 'waiting_for_logo':
            # Handle skipping logo
            if text.lower() in ['skip', 'no logo', 'none', 'no']:
                logger.info(f"User {from_number} chose to skip logo")
                session['logo_image'] = None
                session['state'] = 'waiting_for_details'
                
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "✅ Logo skipped. Now please provide the following details:\n\n"
                    "1️⃣ Company name\n"
                    "2️⃣ Product name\n"
                    "3️⃣ Price\n"
                    "4️⃣ Tagline (optional)\n"
                    "5️⃣ Address (optional)\n\n"
                    "You can send them one by one or all at once."
                )
                return
            else:
                logger.info(f"User {from_number} sent text while waiting for logo")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "I'm waiting for a logo image.\n"
                    "You can send an image file or type 'skip' to continue without a logo."
                )
                return
        
        elif session['state'] == 'waiting_for_details':
            logger.info(f"User {from_number} sent details: {text}")
            # Parse the details
            lines = text.split('\n')
            detail_provided = False
            
            # Check if this is a structured message
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'company' in key:
                        session['details']['company_name'] = value
                        detail_provided = True
                        logger.info(f"Set company_name: {value}")
                    elif 'product' in key:
                        session['details']['product_name'] = value
                        detail_provided = True
                        logger.info(f"Set product_name: {value}")
                    elif 'price' in key:
                        session['details']['price'] = value
                        detail_provided = True
                        logger.info(f"Set price: {value}")
                    elif 'tagline' in key:
                        session['details']['tagline'] = value
                        detail_provided = True
                        logger.info(f"Set tagline: {value}")
                    elif 'address' in key or 'location' in key:
                        session['details']['address'] = value
                        detail_provided = True
                        logger.info(f"Set address: {value}")
            
            # If no structured details, try to interpret single line inputs
            if not detail_provided:
                logger.info("No structured details found, interpreting as single value")
                if not session['details'].get('company_name'):
                    session['details']['company_name'] = text
                    logger.info(f"Set company_name: {text}")
                    detail_provided = True
                elif not session['details'].get('product_name'):
                    session['details']['product_name'] = text
                    logger.info(f"Set product_name: {text}")
                    detail_provided = True
                elif not session['details'].get('price'):
                    session['details']['price'] = text
                    logger.info(f"Set price: {text}")
                    detail_provided = True
                elif not session['details'].get('tagline'):
                    session['details']['tagline'] = text
                    logger.info(f"Set tagline: {text}")
                    detail_provided = True
                elif not session['details'].get('address'):
                    session['details']['address'] = text
                    logger.info(f"Set address: {text}")
                    detail_provided = True
            
            # Send updated status and next step
            logger.info(f"Sending status update to {from_number}")
            status_msg = "📝 Current details:\n\n"
            status_msg += f"Company: {session['details'].get('company_name', '❌')}\n"
            status_msg += f"Product: {session['details'].get('product_name', '❌')}\n"
            status_msg += f"Price: {session['details'].get('price', '❌')}\n"
            status_msg += f"Tagline: {session['details'].get('tagline', '➖')}\n"
            status_msg += f"Address: {session['details'].get('address', '➖')}\n\n"
            
            # Check what's still needed
            if not session['details'].get('company_name'):
                status_msg += "👉 Please send your company name.\n"
            elif not session['details'].get('product_name'):
                status_msg += "👉 Please send your product name.\n"
            elif not session['details'].get('price'):
                status_msg += "👉 Please send the price.\n"
            else:
                status_msg += "✅ All required information received!\n\n"
                status_msg += "To generate the marketing image, send 'generate'\n"
                status_msg += "To add optional details (tagline, address), just send them."
            
            marketing_bot.waapi_client.send_message(from_number, status_msg)
            return
        
        # Default state - waiting for command
        else:
            logger.info(f"User {from_number} sent message in default state")
            marketing_bot.waapi_client.send_message(
                from_number,
                "👋 Welcome to Marketing Image Generator!\n\n"
                "To create a marketing image, send 'edit' to start."
            )
            
    except Exception as e:
        logger.error(f"Error handling text message: {str(e)}")
        try:
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, an error occurred. Please try again.\n"
                "Send 'edit' to start over."
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {str(send_error)}")

def handle_image_message(from_number: str, media_data):
    """Handle incoming image messages"""
    try:
        logger.info(f"Image received from {from_number}. Processing...")
        
        # Create session if not exists
        if from_number not in user_sessions:
            logger.info(f"Creating new session for {from_number}")
            user_sessions[from_number] = {
                "product_image": None,
                "logo_image": None,
                "details": {},
                "state": "waiting_for_command"
            }
        
        session = user_sessions[from_number]
        
        # Debug session state
        logger.info(f"Current session state: {session['state']}")
        
        # Check if we're in the right state to receive an image
        if session['state'] == 'waiting_for_image':
            # Processing product image
            try:
                if media_data and media_data.get('data'):  
                    # Normal path - we have base64 data
                    logger.info("Product image data found. Decoding base64 data...")
                    image_bytes = base64.b64decode(media_data['data'])
                else:
                    # No media data in webhook - this is common with WaAPI's trial limitations
                    # Create a placeholder image for testing
                    logger.warning("No image data in webhook - creating placeholder image")
                    
                    # Save a placeholder image (1x1 transparent pixel)
                    image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
                    
                    # Let the user know about the limitation
                    marketing_bot.waapi_client.send_message(
                        from_number,
                        "⚠️ I received your image but couldn't access its data due to trial account limitations.\n\n"
                        "I'll continue with a placeholder image for testing purposes."
                    )
                
                # Save the product image
                filename = f"product_{from_number.replace('@c.us', '')}_{int(datetime.now().timestamp())}.png"
                image_path = os.path.join("images/input", filename)
                
                logger.info(f"Saving product image to {image_path}")
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"Product image saved to {image_path}")
                session['product_image'] = image_path
                session['state'] = 'waiting_for_logo'
                logger.info("Session state changed to 'waiting_for_logo'")
                
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "✅ Product image received!\n\n"
                    "You can now optionally send a logo image. If you don't want to include a logo, type 'skip'."
                )
                
            except Exception as e:
                logger.error(f"Error processing product image: {str(e)}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your product image. Please try again."
                )
                
        elif session['state'] == 'waiting_for_logo':
            # Processing logo image
            try:
                if media_data and media_data.get('data'):  
                    # Normal path - we have base64 data
                    logger.info("Logo image data found. Decoding base64 data...")
                    image_bytes = base64.b64decode(media_data['data'])
                else:
                    # No media data in webhook
                    logger.warning("No logo image data in webhook - creating placeholder logo")
                    
                    # Save a placeholder image
                    image_bytes = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
                    
                    marketing_bot.waapi_client.send_message(
                        from_number,
                        "⚠️ I received your logo but couldn't access its data due to trial account limitations.\n\n"
                        "I'll continue with a placeholder logo for testing purposes."
                    )
                
                # Save the logo image
                filename = f"logo_{from_number.replace('@c.us', '')}_{int(datetime.now().timestamp())}.png"
                image_path = os.path.join("images/logos", filename)
                
                logger.info(f"Saving logo image to {image_path}")
                os.makedirs("images/logos", exist_ok=True)
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                logger.info(f"Logo image saved to {image_path}")
                session['logo_image'] = image_path
                session['state'] = 'waiting_for_details'
                logger.info("Session state changed to 'waiting_for_details'")
                
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "✅ Logo image received!\n\n"
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
                    "Price: $20"
                )
                
            except Exception as e:
                logger.error(f"Error processing logo image: {str(e)}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "Sorry, I couldn't process your logo image. Please try again or type 'skip' to continue without a logo."
                )
                
        else:
            # Received image in wrong state
            logger.warning(f"Received image but session state is {session['state']}, not expecting an image")
            marketing_bot.waapi_client.send_message(
                from_number,
                "I wasn't expecting an image right now.\n"
                "To start the process, please send 'edit' first.\n\n"
                "You can also type 'cancel' to exit the current process."
            )
            
    except Exception as e:
        logger.error(f"Error handling image message: {str(e)}")
        try:
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, I couldn't process your image. Please try again.\n"
                "Start over by sending 'edit'."
            )
        except Exception as send_error:
            logger.error(f"Failed to send error message: {str(send_error)}")

###################
# FLASK ROUTES
###################

@app.route('/')
def home():
    """Render the home page"""
    logger.info("Home page accessed")
    return jsonify({
        "service": "Marketing Image Generator (WhatsApp Version)",
        "status": "running",
        "version": "1.0",
        "endpoint": "/webhook"
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    logger.debug("Health check accessed")
    return jsonify({"status": "healthy"})

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    logger.debug(f"Serving image: {path}")
    directory, filename = os.path.split(path)
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via WaAPI webhook"""
    try:
        # Extract the webhook data
        webhook_data = request.json
        logger.info(f"Received webhook: {json.dumps(webhook_data)}")
        debug_print(f"Received webhook data: {json.dumps(webhook_data)}")
        
        # Log request details for debugging
        debug_print(f"Request headers: {dict(request.headers)}")
        debug_print(f"Request method: {request.method}")
        debug_print(f"Request content type: {request.content_type}")
        debug_print(f"Request content length: {request.content_length}")
        
        # Extract message ID to prevent duplicate processing
        message_data = webhook_data.get('data', {}).get('message', {})
        
        # Debug the message structure
        logger.debug(f"Message data structure: {json.dumps(message_data)}")
        
        message_id = message_data.get('id', {})
        if isinstance(message_id, dict):
            serialized_id = message_id.get('_serialized', '')
        else:
            serialized_id = str(message_id)
        
        # Skip if we've already processed this message
        if serialized_id and serialized_id in processed_messages:
            logger.info(f"Skipping duplicate message: {serialized_id}")
            return jsonify({"status": "success", "message": "Duplicate message skipped"})
        
        # Mark as processed if it has an ID
        if serialized_id:
            processed_messages[serialized_id] = datetime.now().timestamp()
            logger.debug(f"Added message {serialized_id} to processed cache")
            
            # Limit cache size by removing old entries (keep last 100)
            if len(processed_messages) > 100:
                oldest = sorted(processed_messages.items(), key=lambda x: x[1])[0][0]
                processed_messages.pop(oldest)
                logger.debug(f"Removed oldest message {oldest} from cache")
        
        # Check if this is a message event
        if webhook_data.get('event') in ['message', 'message_create']:
            # Extract message details - safer parsing with fallbacks
            message_type = message_data.get('type', '')
            from_number = message_data.get('from', '')
            # Handle None values properly when parsing body
            raw_body = message_data.get('body')
            body = raw_body.strip().lower() if isinstance(raw_body, str) else ''
            
            # Check for media in multiple ways
            has_media = (
                message_data.get('hasMedia', False) or 
                message_type in ['image', 'sticker', 'video', 'document'] or
                'mediaData' in message_data or
                '_data' in message_data and message_data['_data'].get('type') in ['image', 'sticker', 'video', 'document']
            )
            
            logger.info(f"Received message from {from_number}: {body}, Media: {has_media}, Type: {message_type}")
            
            # Check if the number is in the proper format
            if '@c.us' not in str(from_number):
                logger.error(f"Invalid phone number format: {from_number}")
                return jsonify({"status": "error", "message": "Invalid phone number format"})
            
            # Try to get media data in multiple ways
            media_data = webhook_data.get('data', {}).get('media', {})
            if not media_data:
                logger.debug("No direct media data found, checking alternative locations")
                if '_data' in message_data and message_data['_data'].get('type') in ['image', 'sticker', 'video', 'document']:
                    logger.debug("Found media in _data")
                    # Try to extract media from message_data itself for different webhook formats
                    if hasattr(message_data, 'mediaData') or 'mediaData' in message_data:
                        media_data = message_data.get('mediaData', {})
                    elif hasattr(message_data, 'mimetype') or 'mimetype' in message_data:
                        # It might be directly in the message
                        media_data = message_data
            
            # Handle media messages
            if has_media:
                logger.info(f"Detected media in message from {from_number}, type: {message_type}")
                if not media_data:
                    # If webhook doesn't include media data, try to fetch it
                    logger.info("Media indicated but no media data in webhook, will attempt download")
                
                # Process as image regardless of empty media data - handler will check session state
                handle_image_message(from_number, media_data)
                return jsonify({"status": "success", "message": "Media processed"})
                
            # Handle text messages
            handle_text_message(from_number, body)
            return jsonify({"status": "success", "message": "Text processed"})
        
        # Handle other event types if needed
        logger.debug(f"Event {webhook_data.get('event')} processed")
        return jsonify({"status": "success", "message": "Event processed"})
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 5000))
    
    # Print system information for debugging
    import sys
    import platform
    
    debug_print(f"Python version: {sys.version}")
    debug_print(f"Platform info: {platform.platform()}")
    debug_print(f"Pillow version: {Image.__version__}")
    debug_print(f"Running in directory: {os.getcwd()}")
    debug_print(f"Environment variables: PORT={port}")
    
    # Check for required directories
    for directory in ['images', 'images/input', 'images/output', 'images/masks', 'images/logos']:
        path = os.path.join(os.getcwd(), directory)
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        is_writable = os.access(path, os.W_OK) if exists else False
        debug_print(f"Directory {path}: exists={exists}, is_dir={is_dir}, writable={is_writable}")
    
    # Test OpenAI connectivity
    debug_print("Testing OpenAI connectivity...")
    try:
        models = openai_client.models.list()
        debug_print(f"OpenAI connectivity test successful. Available models count: {len(models.data)}")
        gpt_image_available = any(model.id == "gpt-image-1" for model in models.data)
        debug_print(f"gpt-image-1 model available: {gpt_image_available}")
    except Exception as e:
        debug_print(f"OpenAI connectivity test failed: {str(e)}")
    
    # Run the Flask app
    logger.info(f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    debug_print(f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production