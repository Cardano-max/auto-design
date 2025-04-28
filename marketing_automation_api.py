#!/usr/bin/env python3
"""
Marketing Image Generator - WaAPI WhatsApp Version
--------------------------------------------
Production version with WaAPI WhatsApp API integration
Configured for Railway.com deployment with improved OpenAI integration
"""

import os
import base64
import json
import logging
import traceback
import requests
import time
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("marketing_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)
os.makedirs('images/masks', exist_ok=True)  # For image editing masks

# Store user sessions
user_sessions = {}

# Store processed messages to prevent duplicates
processed_messages = {}

# Store last message time for rate limiting
last_message_time = {}

# Initialize OpenAI with compatibility handling
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required")

# Modern OpenAI client initialization - avoiding problematic parameters
try:
    from openai import OpenAI
    # Initialize without proxies parameter that was causing issues
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_VERSION = "new"
    logger.info("Using new OpenAI client version")
except Exception as e:
    logger.warning(f"Using legacy OpenAI API: {str(e)}")
    import openai
    openai.api_key = OPENAI_API_KEY
    openai_client = openai
    OPENAI_VERSION = "legacy"

# WaAPI configuration
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
        return f"""Create a professional marketing poster for a product with these specifications:

1. PRODUCT PRESENTATION:
   - Place the product as the central focal point
   - Remove the original background completely
   - Apply professional product shadows for dimension

2. BRANDING ELEMENTS:
   - Company name: "{product_details.get('company_name', '')}" positioned at the top in an elegant, readable font
   - Product name: "{product_details.get('product_name', '')}" below the product image in bold typography
   - Price: "{product_details.get('price', '')}" displayed in a circular badge, bottom right corner using maroon/red with white text
   - Include a tagline: "{product_details.get('tagline', '')}" in italic font between company name and product image
   - Location/address: "{product_details.get('address', '')}" in smaller font below the product name

3. DESIGN STYLE:
   - Professional marketing aesthetics similar to high-end brands
   - Clean, modern layout with balanced spacing
   - Background: {product_details.get('background_style', 'light beige')} with subtle texture/gradient
   - Use complementary colors that enhance the product's appeal
   - Ensure all text is perfectly readable

The final image should look like it was created by an expert graphic designer for a premium brand's marketing campaign.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template"""
        logger.debug(f"Generating beverage template with product details: {product_details}")
        return f"""Create a premium café marketing poster for a beverage with these specifications:

1. PRODUCT ENHANCEMENT:
   - Center the beverage as the focal point
   - Remove the original background completely
   - Keep whipped cream, garnishes, and toppings visible and appetizing
   - Add subtle steam effects for hot drinks or condensation for cold drinks
   - Enhance beverage colors for visual appeal

2. BRANDING PLACEMENT:
   - Company: "{product_details.get('company_name', '')}" at top in elegant script font
   - Product: "{product_details.get('product_name', '')}" below the product
   - Price: "{product_details.get('price', '')}" in a circular price tag, bottom corner
   - Tagline: "{product_details.get('tagline', '')}" below company name
   - Location: "{product_details.get('address', '')}" in smaller font at bottom

3. AESTHETIC ELEMENTS:
   - Warm background color that complements the drink
   - Subtle coffee beans/relevant ingredients in background
   - Soft vignette effect to focus on the drink
   - Style similar to Starbucks or premium coffee shop marketing

Create a professional marketing poster as described above.
"""

    @staticmethod
    def get_food_template(product_details: Dict) -> str:
        """Generate food-specific prompt template"""
        logger.debug(f"Generating food template with product details: {product_details}")
        return f"""Create an appetizing marketing poster for a food product with these specifications:

1. PRODUCT PRESENTATION:
   - Center the food item as the focal point
   - Remove the original background completely
   - Enhance food textures and colors for appetite appeal
   - Add steam effects for hot items or fresh appearance for cold items
   - Ensure the food looks perfectly prepared

2. BRANDING ELEMENTS:
   - Company: "{product_details.get('company_name', '')}" at top in clean font
   - Product: "{product_details.get('product_name', '')}" below the food in bold font
   - Price: "{product_details.get('price', '')}" in an eye-catching badge
   - Tagline: "{product_details.get('tagline', '')}" emphasizing flavor/freshness
   - Location: "{product_details.get('address', '')}" at bottom in readable font

3. MARKETING ENHANCEMENT:
   - Background complementing the food's colors
   - Subtle ingredients in background for context
   - Professional food photography style
   - Balance all elements for premium restaurant look

Create a professional food marketing poster as described above.
"""

###################
# IMAGE PROCESSING
###################

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def create_transparent_mask(image_path: str) -> str:
        """
        Create a transparent mask from an image for inpainting
        Returns the path to the created mask
        """
        try:
            logger.info(f"Creating transparent mask for {image_path}")
            
            # Load the original image
            img = Image.open(image_path)
            
            # Convert to RGBA (if not already)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            
            # Create a fully transparent mask (all alpha values set to 0)
            width, height = img.size
            mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            
            # Save the mask
            mask_filename = f"mask_{os.path.basename(image_path)}"
            mask_path = os.path.join("images/masks", mask_filename)
            mask.save(mask_path, format="PNG")
            
            logger.info(f"Transparent mask created at {mask_path}")
            return mask_path
            
        except Exception as e:
            logger.error(f"Error creating transparent mask: {str(e)}")
            return None
    
    @staticmethod
    def resize_image_if_needed(image_path: str, max_size: int = 1024) -> str:
        """
        Resize an image if it's larger than max_size while maintaining aspect ratio
        Returns the path to the resized image
        """
        try:
            logger.info(f"Checking if {image_path} needs resizing")
            
            # Load the image
            img = Image.open(image_path)
            
            # Check if resizing is needed
            width, height = img.size
            if width <= max_size and height <= max_size:
                logger.info(f"Image is already within size limits: {width}x{height}")
                return image_path
            
            # Calculate new dimensions while maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the resized image
            filename, ext = os.path.splitext(os.path.basename(image_path))
            resized_filename = f"{filename}_resized{ext}"
            resized_path = os.path.join(os.path.dirname(image_path), resized_filename)
            
            resized_img.save(resized_path, format="PNG" if ext.lower() == ".png" else "JPEG")
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}, saved at {resized_path}")
            
            return resized_path
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image_path  # Return original path in case of error

###################
# IMAGE GENERATION
###################

class ImageGenerator:
    def __init__(self, api_key: str):
        logger.info("Initializing ImageGenerator")
        self.api_key = api_key
        self.client = openai_client
        self.openai_version = OPENAI_VERSION
        logger.info(f"ImageGenerator initialized with OpenAI API ({self.openai_version} version)")
    
    def generate_marketing_image(self, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using OpenAI API"""
        try:
            product_name = product_details.get('product_name', 'product')
            logger.info(f"Starting image generation for {product_name} as {product_type}")
            logger.info(f"API key length: {len(self.api_key)}")
            
            # Verify that the image file exists
            if not os.path.exists(product_image_path):
                logger.error(f"Product image file not found: {product_image_path}")
                return None
            
            # Resize image if needed (API requirements)
            resized_image_path = ImageProcessor.resize_image_if_needed(product_image_path)
            
            # Select the appropriate prompt template
            if product_type.lower() == "beverage":
                prompt = PromptTemplates.get_beverage_template(product_details)
                logger.info("Using beverage template")
            elif product_type.lower() == "food":
                prompt = PromptTemplates.get_food_template(product_details)
                logger.info("Using food template")
            else:
                prompt = PromptTemplates.get_master_template(product_details)
                logger.info("Using master template")
            
            # Generate the image
            try:
                logger.info("Sending image generation request to OpenAI API")
                
                if self.openai_version == "new":
                    # Modern OpenAI client (dall-e-3)
                    logger.info("Using new OpenAI client for image generation with DALL-E 3")
                    
                    # For new versions, we'll try to use the image edit endpoint first
                    try:
                        # Create a transparent mask for editing
                        mask_path = ImageProcessor.create_transparent_mask(resized_image_path)
                        
                        if mask_path and os.path.exists(mask_path):
                            logger.info(f"Using image editing with mask: {mask_path}")
                            
                            # Open image and mask files
                            with open(resized_image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                                # Use the edit endpoint with DALL-E 3
                                result = self.client.images.edit(
                                    model="dall-e-3",
                                    image=img_file,
                                    mask=mask_file,
                                    prompt=prompt,
                                    size="1024x1024",
                                    n=1
                                )
                                logger.info("OpenAI edit API call successful")
                        else:
                            raise ValueError("Mask creation failed, falling back to generation")
                    
                    except Exception as edit_error:
                        logger.warning(f"Image edit failed: {str(edit_error)}, falling back to generation")
                        
                        # Fallback to standard image generation
                        result = self.client.images.generate(
                            model="dall-e-3",
                            prompt=prompt,
                            size="1024x1024",
                            quality="standard",
                            n=1
                        )
                        logger.info("OpenAI generation API call successful")
                    
                    # Process the response
                    if hasattr(result, 'data') and len(result.data) > 0:
                        if hasattr(result.data[0], 'url'):
                            logger.info("Image URL received, downloading image")
                            image_url = result.data[0].url
                            response = requests.get(image_url)
                            image_bytes = response.content
                            logger.info("Image downloaded successfully")
                        elif hasattr(result.data[0], 'b64_json'):
                            logger.info("Image base64 data received, decoding")
                            image_bytes = base64.b64decode(result.data[0].b64_json)
                            logger.info("Image decoded successfully")
                        else:
                            logger.error("No image URL or base64 data in OpenAI response")
                            return None
                    else:
                        logger.error("No data in OpenAI response")
                        return None
                    
                else:
                    # Legacy OpenAI API
                    logger.info("Using legacy OpenAI client for image generation")
                    try:
                        # Try edit endpoint first (if available)
                        mask_path = ImageProcessor.create_transparent_mask(resized_image_path)
                        
                        if mask_path and os.path.exists(mask_path) and hasattr(self.client.Image, 'create_edit'):
                            logger.info(f"Using legacy image editing with mask: {mask_path}")
                            
                            with open(resized_image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                                result = self.client.Image.create_edit(
                                    image=img_file,
                                    mask=mask_file,
                                    prompt=prompt,
                                    n=1,
                                    size="1024x1024"
                                )
                                logger.info("Legacy OpenAI edit request completed")
                        else:
                            raise ValueError("Mask creation failed or edit not available, using standard generation")
                            
                    except Exception as edit_error:
                        logger.warning(f"Legacy image edit failed: {str(edit_error)}, using standard generation")
                        
                        # Standard image generation (DALL-E 2)
                        result = self.client.Image.create(
                            prompt=prompt,
                            n=1,
                            size="1024x1024"
                        )
                        logger.info("Legacy OpenAI generation completed")
                    
                    # Process the response
                    if 'data' in result and len(result['data']) > 0:
                        if 'url' in result['data'][0]:
                            logger.info("Image URL received, downloading image")
                            image_url = result['data'][0]['url']
                            response = requests.get(image_url)
                            image_bytes = response.content
                            logger.info("Image downloaded successfully")
                        elif 'b64_json' in result['data'][0]:
                            logger.info("Image base64 data received, decoding")
                            image_bytes = base64.b64decode(result['data'][0]['b64_json'])
                            logger.info("Image decoded successfully")
                        else:
                            logger.error("No image URL or base64 data in OpenAI response")
                            return None
                    else:
                        logger.error("No image data in response")
                        return None
                
                # Save the image
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                # Convert to image
                logger.info("Creating image from bytes")
                image = Image.open(BytesIO(image_bytes))
                
                # Optionally resize for optimization
                if image.size[0] > 1500 or image.size[1] > 1500:
                    logger.info(f"Resizing image from {image.size} to max 1500px")
                    image.thumbnail((1500, 1500), Image.LANCZOS)
                
                # Save the image
                logger.info(f"Saving image to {output_path}")
                image.save(output_path, format="PNG", optimize=True)
                
                logger.info(f"Marketing image saved to {output_path}")
                return output_path
                
            except Exception as api_error:
                logger.error(f"OpenAI API Error: {str(api_error)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Try fallback to DALL-E 2
                try:
                    logger.info("Trying fallback to DALL-E 2")
                    if self.openai_version == "new":
                        result = self.client.images.generate(
                            model="dall-e-2",  # Fallback to DALL-E 2
                            prompt=prompt,
                            size="1024x1024",
                            n=1
                        )
                        
                        if hasattr(result.data[0], 'url'):
                            image_url = result.data[0].url
                            response = requests.get(image_url)
                            image_bytes = response.content
                        else:
                            image_bytes = base64.b64decode(result.data[0].b64_json)
                    else:
                        result = self.client.Image.create(
                            prompt=prompt,
                            n=1,
                            size="1024x1024"
                        )
                        
                        if 'url' in result['data'][0]:
                            image_url = result['data'][0]['url']
                            response = requests.get(image_url)
                            image_bytes = response.content
                        else:
                            image_bytes = base64.b64decode(result['data'][0]['b64_json'])
                    
                    # Save the image (same as above)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                    output_filename = f"{product_name_safe}_{timestamp}.png"
                    output_path = os.path.join("images/output", output_filename)
                    
                    image = Image.open(BytesIO(image_bytes))
                    if image.size[0] > 1500 or image.size[1] > 1500:
                        image.thumbnail((1500, 1500), Image.LANCZOS)
                    
                    image.save(output_path, format="PNG", optimize=True)
                    logger.info(f"Fallback image saved to {output_path}")
                    return output_path
                
                except Exception as fallback_error:
                    logger.error(f"Fallback generation failed: {str(fallback_error)}")
                    logger.error(f"Fallback traceback: {traceback.format_exc()}")
                    return None
                
        except Exception as e:
            logger.error(f"Error generating marketing image: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None, retry: bool = True) -> Dict:
        """Make a request to the WaAPI API with retry capability"""
        url = f"{self.api_base_url}/{endpoint}"
        logger.info(f"Making {method} request to {endpoint}")
        
        # Implement retries
        max_retries = 3 if retry else 1
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if method.lower() == "get":
                    response = requests.get(url, headers=self.headers, timeout=30)
                elif method.lower() == "post":