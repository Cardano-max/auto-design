#!/usr/bin/env python3
"""
Marketing Image Generator - WaAPI WhatsApp Version
--------------------------------------------
Production version with WaAPI WhatsApp API integration
Configured for Railway.com deployment with enhanced reliability
"""

import os
import base64
import json
import logging
import traceback
import requests
import time
import threading
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union, Tuple
from PIL import Image, ImageOps
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from dotenv import load_dotenv
import httpx

###################
# CONFIGURATION
###################

# Load environment variables
load_dotenv()

# Create directories for storing images
os.makedirs('images/input', exist_ok=True)
os.makedirs('images/output', exist_ok=True)
os.makedirs('images/masks', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Configure logging with file rotation
log_file = os.path.join('logs', f"marketing_bot_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MarketingBotWhatsApp")

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

# In-memory data stores with thread locks
user_sessions_lock = threading.Lock()
user_sessions = {}

processed_messages_lock = threading.Lock()
processed_messages = {}

last_message_time_lock = threading.Lock()
last_message_time = {}

# Initialize OpenAI with error handling
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_VERSION = "1.55.3"  # Match the version in requirements.txt
openai_client = None

if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY is required for marketing image generation")

try:
    from openai import OpenAI
    
    # Create a custom HTTP client that handles proxy settings
    class CustomHTTPClient(httpx.Client):
        def __init__(self, *args, **kwargs):
            kwargs.pop("proxies", None)  # Remove the 'proxies' argument if present
            super().__init__(*args, **kwargs)
    
    # Initialize with custom HTTP client
    openai_client = OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=CustomHTTPClient(),
        timeout=30.0,
        max_retries=3
    )
    logger.info("Successfully initialized OpenAI client")
except Exception as e:
    logger.critical(f"Failed to initialize OpenAI client: {str(e)}")
    raise RuntimeError(f"Failed to initialize OpenAI: {str(e)}")

# WaAPI configuration with validation
WAAPI_API_TOKEN = os.getenv('WAAPI_API_TOKEN')
WAAPI_INSTANCE_ID = os.getenv('WAAPI_INSTANCE_ID')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
APP_URL = os.getenv('RAILWAY_STATIC_URL', os.getenv('APP_URL', 'http://localhost:8080'))

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
logger.info(f"Application URL: {APP_URL}")

###################
# UTILITIES
###################

def safe_json_loads(json_str: str, default=None) -> Any:
    """Safely load JSON string without raising exceptions"""
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error parsing JSON: {str(e)}")
        return default

def safe_get(dictionary: Dict, key_path: str, default=None) -> Any:
    """
    Safely get value from nested dictionaries using dot notation
    Example: safe_get(data, 'user.profile.name', 'Unknown')
    """
    if not dictionary or not isinstance(dictionary, dict):
        return default
    
    keys = key_path.split('.')
    result = dictionary
    
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    
    return result

def generate_timestamp(format_str: str = "%Y%m%d%H%M%S") -> str:
    """Generate a formatted timestamp string"""
    return datetime.now().strftime(format_str)

def calculate_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA-256 hash of a file for verification"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found for hashing: {file_path}")
            return None
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating file hash: {str(e)}")
        return None

###################
# PROMPT TEMPLATES
###################

class PromptTemplates:
    @staticmethod
    def get_master_template(product_details: Dict) -> str:
        """Generate the master prompt template with product details"""
        logger.debug(f"Generating master template with product details")
        
        # Default values for empty fields
        company = product_details.get('company_name', 'Brand')
        product = product_details.get('product_name', 'Product')
        price = product_details.get('price', '$0.00')
        tagline = product_details.get('tagline', 'Quality Product')
        address = product_details.get('address', '')
        background = product_details.get('background_style', 'light beige')
        
        return f"""Create a professional marketing poster for a product with these specifications:

1. PRODUCT PRESENTATION:
   - Place the product as the central focal point
   - Remove the original background completely
   - Apply professional product shadows for dimension

2. BRANDING ELEMENTS:
   - Company name: "{company}" positioned at the top in an elegant, readable font
   - Product name: "{product}" below the product image in bold typography
   - Price: "{price}" displayed in a circular badge, bottom right corner using maroon/red with white text
   - Include a tagline: "{tagline}" in italic font between company name and product image
   - Location/address: "{address}" in smaller font below the product name

3. DESIGN STYLE:
   - Professional marketing aesthetics similar to high-end brands
   - Clean, modern layout with balanced spacing
   - Background: {background} with subtle texture/gradient
   - Use complementary colors that enhance the product's appeal
   - Ensure all text is perfectly readable

The final image should look like it was created by an expert graphic designer for a premium brand's marketing campaign.
"""

    @staticmethod
    def get_beverage_template(product_details: Dict) -> str:
        """Generate beverage-specific prompt template"""
        logger.debug(f"Generating beverage template with product details")
        
        # Default values for empty fields
        company = product_details.get('company_name', 'Café')
        product = product_details.get('product_name', 'Premium Coffee')
        price = product_details.get('price', '$0.00')
        tagline = product_details.get('tagline', 'Finest Quality')
        address = product_details.get('address', '')
        
        return f"""Create a premium café marketing poster for a beverage with these specifications:

1. PRODUCT ENHANCEMENT:
   - Center the beverage as the focal point
   - Remove the original background completely
   - Keep whipped cream, garnishes, and toppings visible and appetizing
   - Add subtle steam effects for hot drinks or condensation for cold drinks
   - Enhance beverage colors for visual appeal

2. BRANDING PLACEMENT:
   - Company: "{company}" at top in elegant script font
   - Product: "{product}" below the product
   - Price: "{price}" in a circular price tag, bottom corner
   - Tagline: "{tagline}" below company name
   - Location: "{address}" in smaller font at bottom

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
        logger.debug(f"Generating food template with product details")
        
        # Default values for empty fields
        company = product_details.get('company_name', 'Restaurant')
        product = product_details.get('product_name', 'Delicious Dish')
        price = product_details.get('price', '$0.00')
        tagline = product_details.get('tagline', 'Fresh & Delicious')
        address = product_details.get('address', '')
        
        return f"""Create an appetizing marketing poster for a food product with these specifications:

1. PRODUCT PRESENTATION:
   - Center the food item as the focal point
   - Remove the original background completely
   - Enhance food textures and colors for appetite appeal
   - Add steam effects for hot items or fresh appearance for cold items
   - Ensure the food looks perfectly prepared

2. BRANDING ELEMENTS:
   - Company: "{company}" at top in clean font
   - Product: "{product}" below the food in bold font
   - Price: "{price}" in an eye-catching badge
   - Tagline: "{tagline}" emphasizing flavor/freshness
   - Location: "{address}" at bottom in readable font

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
    def create_transparent_mask(image_path: str) -> Optional[str]:
        """
        Create a transparent mask from an image for inpainting
        Returns the path to the created mask
        """
        try:
            logger.info(f"Creating transparent mask for {image_path}")
            
            # Verify image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
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
            
            # Verify the mask was created
            if not os.path.exists(mask_path):
                logger.error(f"Failed to save mask at {mask_path}")
                return None
                
            logger.info(f"Transparent mask created at {mask_path}")
            return mask_path
            
        except Exception as e:
            logger.error(f"Error creating transparent mask: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    @staticmethod
    def resize_image_if_needed(image_path: str, max_size: int = 1024) -> Optional[str]:
        """
        Resize an image if it's larger than max_size while maintaining aspect ratio
        Returns the path to the resized image
        """
        try:
            logger.info(f"Checking if {image_path} needs resizing")
            
            # Verify image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load the image
            img = Image.open(image_path)
            
            # Check image integrity
            try:
                img.verify()
                img = Image.open(image_path)  # Reopen after verify
            except Exception as img_err:
                logger.warning(f"Image verification failed: {str(img_err)}")
                # Try to repair the image
                try:
                    img = Image.open(image_path)
                    img.load()
                    logger.info("Image loaded despite verification failure")
                except Exception as repair_err:
                    logger.error(f"Image repair failed: {str(repair_err)}")
                    return None
            
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
            
            # Determine format based on extension
            save_format = "PNG" if ext.lower() in ['.png', '.webp'] else "JPEG"
            
            # Save with quality parameter
            if save_format == "JPEG":
                resized_img.save(resized_path, format=save_format, quality=90, optimize=True)
            else:
                resized_img.save(resized_path, format=save_format, optimize=True)
                
            logger.info(f"Image resized from {width}x{height} to {new_width}x{new_height}, saved at {resized_path}")
            
            # Verify the resized image was created
            if not os.path.exists(resized_path):
                logger.error(f"Failed to save resized image at {resized_path}")
                return image_path
                
            return resized_path
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            logger.error(traceback.format_exc())
            return image_path  # Return original path in case of error
    
    @staticmethod
    def create_placeholder_image(width: int = 512, height: int = 512, color: tuple = (100, 149, 237)) -> BytesIO:
        """Create a placeholder image when real image data is unavailable"""
        try:
            logger.info(f"Creating placeholder image of size {width}x{height}")
            
            # Create a solid color image
            placeholder_img = Image.new('RGB', (width, height), color)
            
            # Add some visual elements to make it look less like a placeholder
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(placeholder_img)
            
            # Add pattern
            for i in range(0, width, 30):
                draw.line([(i, 0), (i, height)], fill=(color[0]-20, color[1]-20, color[2]-20), width=1)
            for i in range(0, height, 30):
                draw.line([(0, i), (width, i)], fill=(color[0]-20, color[1]-20, color[2]-20), width=1)
                
            # Add text if possible
            try:
                # Try to use a built-in font, fall back to default if not available
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except:
                    font = ImageFont.load_default()
                    
                draw.text((width//2-60, height//2), "Placeholder Image", fill=(255, 255, 255), font=font)
            except Exception as font_err:
                logger.warning(f"Couldn't add text to placeholder: {str(font_err)}")
            
            # Save to BytesIO
            img_buffer = BytesIO()
            placeholder_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            logger.info("Placeholder image created successfully")
            return img_buffer
            
        except Exception as e:
            logger.error(f"Error creating placeholder image: {str(e)}")
            # Create a very basic fallback
            img_buffer = BytesIO()
            Image.new('RGB', (512, 512), (100, 100, 100)).save(img_buffer, format="PNG")
            img_buffer.seek(0)
            return img_buffer

###################
# IMAGE GENERATION
###################

class ImageGenerator:
    def __init__(self, api_key: str):
        logger.info("Initializing ImageGenerator")
        self.api_key = api_key
        self.client = openai_client
        self.openai_version = OPENAI_VERSION  # Add the version attribute
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        logger.info("ImageGenerator initialized with OpenAI API")
    
    def generate_marketing_image(self, 
                                product_image_path: str, 
                                product_details: Dict, 
                                product_type: str = "beverage") -> Optional[str]:
        """Generate a marketing image using OpenAI API with robust error handling"""
        # Input validation
        if not product_image_path or not os.path.exists(product_image_path):
            logger.error(f"Product image not found at: {product_image_path}")
            return None
            
        if not product_details:
            logger.error("No product details provided")
            return None
            
        # Required fields check
        required_fields = ['company_name', 'product_name', 'price']
        missing_fields = [field for field in required_fields if not product_details.get(field)]
        if missing_fields:
            logger.error(f"Missing required product details: {', '.join(missing_fields)}")
            return None
        
        # Attempt image generation with retries
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Starting image generation attempt {attempt}/{self.max_retries}")
                product_name = product_details.get('product_name', 'product')
                logger.info(f"Generating image for {product_name} as {product_type}")
                
                # Verify API key (basic check)
                if not self.api_key or len(self.api_key) < 20:
                    logger.error("Invalid or missing OpenAI API key")
                    return None
                
                # Resize image if needed
                resized_image_path = ImageProcessor.resize_image_if_needed(product_image_path)
                if not resized_image_path:
                    logger.error("Image resizing failed")
                    return None
                
                # Select appropriate prompt template
                if product_type.lower() == "beverage":
                    prompt = PromptTemplates.get_beverage_template(product_details)
                    logger.info("Using beverage template")
                elif product_type.lower() == "food":
                    prompt = PromptTemplates.get_food_template(product_details)
                    logger.info("Using food template")
                else:
                    prompt = PromptTemplates.get_master_template(product_details)
                    logger.info("Using master template")
                
                # Generate image - try DALL-E 3 with advanced handling
                result = self._generate_image_with_dalle3(resized_image_path, prompt)
                
                # If DALL-E 3 failed, try DALL-E 2
                if not result:
                    logger.warning("DALL-E 3 generation failed, falling back to DALL-E 2")
                    result = self._generate_image_with_dalle2(prompt)
                
                # If all attempts failed
                if not result:
                    logger.error("All image generation methods failed")
                    if attempt < self.max_retries:
                        logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                        continue
                    return None
                
                # Save the generated image
                image_bytes, source = result
                timestamp = generate_timestamp()
                product_name_safe = product_details.get('product_name', 'product').replace(' ', '_')[:20]
                output_filename = f"{product_name_safe}_{timestamp}.png"
                output_path = os.path.join("images/output", output_filename)
                
                # Process and save the image
                success = self._save_generated_image(image_bytes, output_path)
                if success:
                    logger.info(f"Marketing image saved to {output_path} using {source}")
                    return output_path
                
                # If saving failed
                if attempt < self.max_retries:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                    continue
                
                return None
                
            except Exception as e:
                logger.error(f"Error in generate_marketing_image (attempt {attempt}): {str(e)}")
                logger.error(traceback.format_exc())
                
                if attempt < self.max_retries:
                    logger.info(f"Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error("All retry attempts failed")
                    return None
        
        return None
    
    def _generate_image_with_dalle3(self, image_path: str, prompt: str) -> Optional[Tuple[bytes, str]]:
        """Try to generate an image using DALL-E 3 with the best available method"""
        try:
            logger.info("Attempting image generation with DALL-E 3")
            
            # Create a transparent mask
            mask_path = ImageProcessor.create_transparent_mask(image_path)
            
            if mask_path and os.path.exists(mask_path):
                logger.info(f"Using DALL-E 3 image editing with mask")
                
                # Open image and mask files
                with open(image_path, "rb") as img_file, open(mask_path, "rb") as mask_file:
                    # Use the edit endpoint with DALL-E 3
                    result = self.client.images.edit(
                        model="dall-e-3",
                        image=img_file,
                        mask=mask_file,
                        prompt=prompt,
                        size="1024x1024",
                        n=1
                    )
                    logger.info("DALL-E 3 edit API call successful")
                
                # Process result
                if hasattr(result, 'data') and len(result.data) > 0:
                    if hasattr(result.data[0], 'url'):
                        logger.info("Image URL received from DALL-E 3 edit")
                        response = requests.get(result.data[0].url, timeout=30)
                        return response.content, "DALL-E 3 Edit"
                    elif hasattr(result.data[0], 'b64_json'):
                        logger.info("Base64 data received from DALL-E 3 edit")
                        image_bytes = base64.b64decode(result.data[0].b64_json)
                        return image_bytes, "DALL-E 3 Edit"
            
            # Try standard DALL-E 3 generation
            logger.info("Using DALL-E 3 standard generation")
            result = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            
            # Process result
            if hasattr(result, 'data') and len(result.data) > 0:
                if hasattr(result.data[0], 'url'):
                    logger.info("Image URL received from DALL-E 3 generation")
                    response = requests.get(result.data[0].url, timeout=30)
                    return response.content, "DALL-E 3 Generation"
                elif hasattr(result.data[0], 'b64_json'):
                    logger.info("Base64 data received from DALL-E 3 generation")
                    image_bytes = base64.b64decode(result.data[0].b64_json)
                    return image_bytes, "DALL-E 3 Generation"
            
            return None
            
        except Exception as e:
            logger.error(f"Error in _generate_image_with_dalle3: {str(e)}")
            return None
    
    def _generate_image_with_dalle2(self, prompt: str) -> Optional[Tuple[bytes, str]]:
        """Generate an image using DALL-E 2 as a fallback"""
        try:
            logger.info("Attempting image generation with DALL-E 2")
            
            result = self.client.images.generate(
                model="dall-e-2",
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            
            # Process result
            if hasattr(result, 'data') and len(result.data) > 0:
                if hasattr(result.data[0], 'url'):
                    logger.info("Image URL received from DALL-E 2")
                    response = requests.get(result.data[0].url, timeout=30)
                    return response.content, "DALL-E 2"
                elif hasattr(result.data[0], 'b64_json'):
                    logger.info("Base64 data received from DALL-E 2")
                    image_bytes = base64.b64decode(result.data[0].b64_json)
                    return image_bytes, "DALL-E 2"
            
            return None
            
        except Exception as e:
            logger.error(f"Error in _generate_image_with_dalle2: {str(e)}")
            return None
    
    def _save_generated_image(self, image_bytes: bytes, output_path: str) -> bool:
        """Process and save generated image with validation"""
        try:
            logger.info(f"Processing and saving image to {output_path}")
            
            # Convert to image and validate
            image = Image.open(BytesIO(image_bytes))
            
            # Check dimensions
            width, height = image.size
            logger.info(f"Image dimensions: {width}x{height}")
            
            # Check if image is valid
            try:
                image.verify()
                # Re-open after verify
                image = Image.open(BytesIO(image_bytes))
            except Exception as img_err:
                logger.warning(f"Image verification failed: {str(img_err)}")
            
            # Optionally resize for optimization
            if width > 1500 or height > 1500:
                logger.info(f"Resizing image from {width}x{height} to max 1500px")
                image.thumbnail((1500, 1500), Image.LANCZOS)
            
            # Save the image with optimization
            image.save(output_path, format="PNG", optimize=True)
            
            # Validate saved file
            if not os.path.exists(output_path):
                logger.error(f"Failed to save image at {output_path}")
                return False
            
            file_size = os.path.getsize(output_path)
            logger.info(f"Image saved successfully. Size: {file_size} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving generated image: {str(e)}")
            logger.error(traceback.format_exc())
            return False

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
        # Trial instance limits: 10 actions per 5 minutes
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.rate_limit_window = 300  # 5 minutes in seconds (trial limit)
        self.rate_limit_count = 0
        self.rate_limit_start = time.time()
        self.rate_limit_max = 8  # Conservative limit (10 - 2 for safety)
        self.rate_limit_backoff = 2.0  # More aggressive backoff
        self.connection_status = "disconnected"
        self.last_status_check = 0
        self.status_check_interval = 60  # seconds (reduced frequency)
        self.trial_number = "923114909725@c.us"  # Your trial number
        self.webhook_retry_count = 0
        self.max_webhook_retries = 3
        logger.info("WaAPIClient initialized successfully")

    def _check_rate_limit(self) -> bool:
        """Check if we're rate limited and wait if necessary"""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.rate_limit_start > self.rate_limit_window:
            self.rate_limit_count = 0
            self.rate_limit_start = current_time
            return True
            
        # Check if we're rate limited
        if self.rate_limit_count >= self.rate_limit_max:
            wait_time = self.rate_limit_window - (current_time - self.rate_limit_start)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                self.rate_limit_count = 0
                self.rate_limit_start = time.time()
            return True
            
        return False

    def _make_request(self, method: str, endpoint: str, data: Dict = None, retry: bool = True) -> Dict:
        """Make a request to the WaAPI API with improved rate limiting and error handling"""
        url = f"{self.api_base_url}/{endpoint}"
        logger.info(f"Making {method} request to {endpoint}")
        
        # Check rate limit before making request
        if self._check_rate_limit():
            logger.warning("Request delayed due to rate limiting")
        
        # Implement retries with exponential backoff
        max_retries = self.max_retries if retry else 1
        retry_count = 0
        current_delay = self.retry_delay
        
        while retry_count < max_retries:
            try:
                # Prepare request based on method
                if method.lower() == "get":
                    response = requests.get(url, headers=self.headers, timeout=30)
                elif method.lower() == "post":
                    logger.debug(f"POST data: {json.dumps(data) if data else 'None'}")
                    response = requests.post(url, headers=self.headers, json=data, timeout=30)
                elif method.lower() == "put":
                    response = requests.put(url, headers=self.headers, json=data, timeout=30)
                elif method.lower() == "delete":
                    response = requests.delete(url, headers=self.headers, timeout=30)
                else:
                    logger.error(f"Invalid method: {method}")
                    return {"success": False, "error": "Invalid method"}
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Increment rate limit counter
                self.rate_limit_count += 1
                
                # Handle empty responses
                if not response.text.strip():
                    logger.warning(f"Empty response received from {endpoint}")
                    return {"success": False, "error": "Empty response from API"}
                
                # Try to parse JSON response
                try:
                    result = response.json()
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSON parsing error: {str(json_err)}")
                    logger.error(f"Response content: {response.text[:500]}")
                    return {"success": False, "error": f"Invalid JSON response: {str(json_err)}"}
                
                logger.debug(f"Response: {json.dumps(result)}")
                return result
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout to {endpoint}, retry {retry_count+1}/{max_retries}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(current_delay)
                    current_delay *= self.rate_limit_backoff
                
            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 429:  # Rate limiting
                    logger.warning(f"Rate limited by API (429). Retry {retry_count+1}/{max_retries}")
                    # Update rate limit tracking with exponential backoff
                    self.rate_limit_count = self.rate_limit_max
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(current_delay)
                        current_delay *= self.rate_limit_backoff
                elif response.status_code >= 500:  # Server errors
                    logger.warning(f"Server error: {response.status_code}. Retry {retry_count+1}/{max_retries}")
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(current_delay)
                        current_delay *= self.rate_limit_backoff
                else:
                    logger.error(f"HTTP error: {http_err}")
                    return {"success": False, "error": f"HTTP error: {http_err}"}
                    
            except requests.exceptions.RequestException as e:
                if retry_count + 1 < max_retries:
                    logger.warning(f"Request error: {str(e)}, retry {retry_count+1}/{max_retries}")
                    retry_count += 1
                    time.sleep(current_delay)
                    current_delay *= self.rate_limit_backoff
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {str(e)}")
                    return {"success": False, "error": str(e)}
                    
            except Exception as e:
                logger.error(f"Unexpected error making request to WaAPI: {str(e)}")
                logger.error(traceback.format_exc())
                return {"success": False, "error": str(e)}
        
        # If we've exhausted all retries
        return {"success": False, "error": "Request failed after multiple attempts"}

    def get_instance_status(self) -> Dict:
        """Get the status of the instance with caching"""
        current_time = time.time()
        
        # Return cached status if within check interval
        if current_time - self.last_status_check < self.status_check_interval:
            return {
                "success": True,
                "status": self.connection_status,
                "cached": True
            }
        
        logger.info("Checking instance status")
        result = self._make_request("get", f"instances/{self.instance_id}/client/status")
        
        # Process and enhance the result
        if result.get("status") == "success":
            logger.info("Instance status check successful")
            self.connection_status = "connected" if safe_get(result, "data.instance.connected", False) else "disconnected"
            self.last_status_check = current_time
            return {
                "success": True,
                "status": self.connection_status,
                "data": result.get("data", {})
            }
        else:
            logger.error(f"Failed to get instance status: {result.get('error', 'Unknown error')}")
            return {"success": False, "error": result.get("error", "Unknown error")}

    def send_message(self, to: str, message: str, message_type: str = "text") -> Dict:
        """Send a message via WhatsApp with improved validation and rate limiting"""
        # Input validation
        if not to or not message:
            logger.error("Missing required parameters (to or message)")
            return {"success": False, "error": "Missing required parameters"}
            
        # Format phone number to WhatsApp format
        to = self._format_phone_number(to)
        
        # Trial instance validation
        if to != self.trial_number:
            logger.error(f"Trial instance can only send to {self.trial_number}")
            return {"success": False, "error": "Trial instance can only send to registered number"}
        
        # Check instance status before sending
        status = self.get_instance_status()
        if not status.get("success") or status.get("status") != "connected":
            logger.error("Instance is not connected")
            return {"success": False, "error": "Instance is not connected"}
        
        # Apply rate limiting
        with last_message_time_lock:
            current_time = time.time()
            if to in last_message_time and current_time - last_message_time[to] < 2:
                wait_time = 2 - (current_time - last_message_time[to])
                logger.info(f"Rate limiting message to {to} - waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Prepare request data based on message type
        data = {
            "chatId": to
        }
        
        if message_type == "text":
            data["message"] = message
        elif message_type == "template":
            data["template"] = message
        elif message_type == "interactive":
            data["interactive"] = message
        else:
            logger.error(f"Invalid message type: {message_type}")
            return {"success": False, "error": f"Invalid message type: {message_type}"}
        
        # Log message (truncated for privacy)
        display_message = message[:50] + ('...' if len(message) > 50 else '')
        logger.info(f"Sending {message_type} message to {to}: {display_message}")
        
        # Make the API request
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-message", 
            data
        )
        
        # Process the result
        if result.get("status") == "success":
            logger.info(f"Message sent successfully to {to}")
            # Update rate limiting
            with last_message_time_lock:
                last_message_time[to] = time.time()
            return {"success": True, "data": result.get("data", {})}
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Error sending message: {error_msg}")
            return {"success": False, "error": error_msg}

    def send_template_message(self, to: str, template_name: str, template_variables: Dict = None) -> Dict:
        """Send a template message with variables"""
        if not template_name:
            logger.error("Template name is required")
            return {"success": False, "error": "Template name is required"}
            
        template_data = {
            "name": template_name,
            "variables": template_variables or {}
        }
        
        return self.send_message(to, json.dumps(template_data), "template")

    def send_interactive_message(self, to: str, interactive_data: Dict) -> Dict:
        """Send an interactive message (buttons, lists, etc.)"""
        if not interactive_data:
            logger.error("Interactive data is required")
            return {"success": False, "error": "Interactive data is required"}
            
        return self.send_message(to, json.dumps(interactive_data), "interactive")

    def send_media(self, to: str, caption: str = "", media_url: str = None, media_base64: str = None, 
                  filename: str = None, is_sticker: bool = False) -> Dict:
        """Send a media message via WhatsApp with validation and fallbacks"""
        # Input validation
        if not to:
            logger.error("Missing recipient parameter (to)")
            return {"success": False, "error": "Missing recipient parameter"}
            
        if not media_url and not media_base64:
            logger.error("No media provided (URL or base64)")
            return {"success": False, "error": "No media provided"}
        
        # Format phone number to WhatsApp format
        to = self._format_phone_number(to)
        
        # Apply rate limiting
        with last_message_time_lock:
            current_time = time.time()
            if to in last_message_time and current_time - last_message_time[to] < 2:
                wait_time = 2 - (current_time - last_message_time[to])
                logger.info(f"Rate limiting media message to {to} - waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
        
        # Prepare request data
        data = {
            "chatId": to,
            "mediaCaption": caption or ""
        }
        
        # Add either URL or base64 data
        if media_url:
            logger.info(f"Sending media from URL to {to}")
            data["mediaUrl"] = media_url
            
            # Verify URL is accessible
            try:
                head_response = requests.head(media_url, timeout=10)
                if head_response.status_code != 200:
                    logger.warning(f"Media URL may not be accessible: {head_response.status_code}")
            except Exception as url_check_err:
                logger.warning(f"Failed to verify media URL: {str(url_check_err)}")
            
        elif media_base64:
            logger.info(f"Sending media from base64 to {to}")
            
            # Validate base64 data
            try:
                base64.b64decode(media_base64)
            except Exception as b64_err:
                logger.error(f"Invalid base64 data: {str(b64_err)}")
                return {"success": False, "error": "Invalid base64 data"}
            
            data["mediaBase64"] = media_base64
            
            # Filename is required for base64 media
            if not filename:
                filename = f"media_{int(datetime.now().timestamp())}.png"
            data["mediaName"] = filename
        
        # Set if image should be sent as sticker
        if is_sticker:
            data["asSticker"] = True
            
        # Make API request
        result = self._make_request(
            "post", 
            f"instances/{self.instance_id}/client/action/send-media", 
            data
        )
        
        # Process the result
        if result.get("status") == "success":
            logger.info(f"Media sent successfully to {to}")
            # Update rate limiting
            with last_message_time_lock:
                last_message_time[to] = time.time()
            return {"success": True, "data": result.get("data", {})}
        else:
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Error sending media: {error_msg}")
            
            # Try to recover from common errors
            if "base64" in str(error_msg).lower() and media_url:
                logger.warning("Media URL error, attempting to download and send as base64")
                try:
                    response = requests.get(media_url, timeout=30)
                    if response.status_code == 200:
                        media_bytes = response.content
                        media_b64 = base64.b64encode(media_bytes).decode('utf-8')
                        return self.send_media(to, caption, None, media_b64, filename, is_sticker)
                except Exception as recovery_err:
                    logger.error(f"Recovery attempt failed: {str(recovery_err)}")
            
            return {"success": False, "error": error_msg}
    
    def _format_phone_number(self, phone_number: str) -> str:
        """Format phone number to the expected format for WhatsApp"""
        if not phone_number:
            return ""
            
        logger.debug(f"Formatting phone number: {phone_number}")
        
        # Already in WhatsApp format
        if '@c.us' in phone_number:
            return phone_number
        
        # Remove any non-numeric characters
        clean_number = ''.join(filter(str.isdigit, phone_number))
        
        # Validate
        if not clean_number:
            logger.error(f"Invalid phone number: {phone_number}")
            return ""
        
        # Format for WhatsApp: number@c.us
        result = f"{clean_number}@c.us"
        logger.debug(f"Formatted phone number: {result}")
        return result
    
    def is_registered_user(self, phone_number: str) -> bool:
        """Check if a phone number is registered on WhatsApp"""
        # Validate input
        if not phone_number:
            logger.error("Empty phone number provided")
            return False
            
        # Format the phone number
        contact_id = self._format_phone_number(phone_number)
        if not contact_id:
            logger.error(f"Invalid phone number format: {phone_number}")
            return False
            
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
            is_registered = safe_get(result, "data.isRegisteredUser", False)
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
        try:
            self.image_generator = ImageGenerator(openai_key)
            self.waapi_client = WaAPIClient(waapi_token, waapi_instance_id)
            self.max_retries = 3
            self.retry_delay = 5  # seconds
            logger.info("MarketingBot initialized with OpenAI and WaAPI")
        except Exception as e:
            logger.critical(f"Failed to initialize MarketingBot: {str(e)}")
            logger.critical(traceback.format_exc())
            raise RuntimeError(f"MarketingBot initialization failed: {str(e)}")
    
    def process_request(self, user_id: str, product_image_path: str, product_details: Dict, product_type: str = "beverage") -> Dict:
        """Process a marketing image request with comprehensive error handling"""
        request_id = f"{user_id}_{int(time.time())}"
        logger.info(f"[{request_id}] Processing request for user {user_id}")
        
        try:
            # Input validation with detailed logging
            if not user_id:
                logger.error(f"[{request_id}] Missing user ID")
                return {"success": False, "error": "Missing user ID"}
                
            logger.info(f"[{request_id}] Product image path: {product_image_path}")
            logger.info(f"[{request_id}] Product details: {json.dumps(product_details)}")
            logger.info(f"[{request_id}] Product type: {product_type}")
            
            # Verify image path existence
            if not product_image_path:
                logger.error(f"[{request_id}] No product image path provided")
                return {"success": False, "error": "No product image path provided"}
                
            if not os.path.exists(product_image_path):
                logger.error(f"[{request_id}] Product image not found at path: {product_image_path}")
                return {"success": False, "error": "Product image file not found"}
                
            # Validate image file
            try:
                with Image.open(product_image_path) as img:
                    width, height = img.size
                    format = img.format
                    logger.info(f"[{request_id}] Image validated: {width}x{height} {format}")
            except Exception as img_err:
                logger.error(f"[{request_id}] Invalid image file: {str(img_err)}")
                return {"success": False, "error": "Invalid image file"}
            
            # Verify product details
            if not product_details:
                logger.error(f"[{request_id}] No product details provided")
                return {"success": False, "error": "No product details provided"}
                
            required_fields = ['company_name', 'product_name', 'price']
            missing_fields = [field for field in required_fields if not product_details.get(field)]
            if missing_fields:
                fields_str = ', '.join(missing_fields)
                logger.error(f"[{request_id}] Missing required product details: {fields_str}")
                return {"success": False, "error": f"Missing required details: {fields_str}"}
            
            # Update session state to generating
            update_session(user_id, {'state': 'generating_image'})
            
            # Attempt image generation with retries
            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.info(f"[{request_id}] Starting image generation attempt {attempt}/{self.max_retries}")
                    
                    # Generate image - try DALL-E 3 first
                    result = self.image_generator.generate_marketing_image(
                        product_image_path,
                        product_details,
                        product_type
                    )
                    
                    if result and result.get('success'):
                        # Update session state to delivering
                        update_session(user_id, {'state': 'delivering_image'})
                        
                        # Get image path and URL
                        image_path = result.get('image_path')
                        image_url = result.get('image_url')
                        
                        # Send the image to user
                        try:
                            # Read the image for base64 encoding
                            with open(image_path, 'rb') as img_file:
                                img_data = img_file.read()
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                            
                            # Send the generated image
                            media_result = self.waapi_client.send_media(
                                user_id,
                                "🎉 Here's your marketing image!\n\n"
                                "To create another image, send 'edit' again.",
                                media_base64=img_base64,
                                filename=os.path.basename(image_path)
                            )
                            
                            if media_result.get('success'):
                                logger.info(f"[{request_id}] Image sent successfully")
                                # Reset session state
                                update_session(user_id, {'state': 'waiting_for_command'})
                                return result
                            else:
                                # Try alternate approach - send image URL if available
                                if image_url:
                                    logger.info(f"[{request_id}] Trying to send image URL as fallback")
                                    self.waapi_client.send_message(
                                        user_id,
                                        f"I created your marketing image but couldn't send it directly.\n\n"
                                        f"You can view and download it here:\n{image_url}\n\n"
                                        f"To create another image, send 'edit' again."
                                    )
                                    # Reset session state
                                    update_session(user_id, {'state': 'waiting_for_command'})
                                    return result
                                else:
                                    raise Exception("Failed to send image and no URL available")
                            
                        except Exception as send_error:
                            logger.error(f"[{request_id}] Error sending image: {str(send_error)}")
                            if attempt < self.max_retries:
                                logger.info(f"[{request_id}] Retrying in {self.retry_delay} seconds...")
                                time.sleep(self.retry_delay)
                                continue
                            else:
                                raise
                    
                    # If generation failed, try fallback
                    if attempt < self.max_retries:
                        logger.warning(f"[{request_id}] Generation failed, retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"[{request_id}] All generation attempts failed")
                        return {"success": False, "error": "Failed to generate image after multiple attempts"}
                        
                except Exception as e:
                    logger.error(f"[{request_id}] Error in generate_marketing_image (attempt {attempt}): {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    if attempt < self.max_retries:
                        logger.info(f"[{request_id}] Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                    else:
                        logger.error(f"[{request_id}] All retry attempts failed")
                        return {"success": False, "error": str(e)}
            
            return {"success": False, "error": "Unknown error occurred"}
            
        except Exception as e:
            logger.error(f"[{request_id}] Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            # Reset session state on error
            update_session(user_id, {'state': 'waiting_for_command'})
            return {"success": False, "error": str(e)}

    def verify_api_connections(self) -> Dict:
        """Verify connections to all required APIs"""
        results = {
            "whatsapp": False,
            "openai": False,
            "overall": False
        }
        
        # Check WhatsApp API connection
        try:
            logger.info("Verifying WhatsApp API connection")
            status_result = self.waapi_client.get_instance_status()
            results["whatsapp"] = status_result.get("success", False)
            results["whatsapp_details"] = status_result.get("status", "unknown")
            logger.info(f"WhatsApp API connection: {results['whatsapp']}")
        except Exception as wa_err:
            logger.error(f"WhatsApp API verification failed: {str(wa_err)}")
            results["whatsapp_error"] = str(wa_err)
        
        # Check OpenAI API connection
        try:
            logger.info("Verifying OpenAI API connection")
            
            # Simple test based on client version
            if self.image_generator.openai_version == "new":
                # New client test
                result = self.image_generator.client.models.list()
                results["openai"] = True
                results["openai_details"] = "Models list retrieved"
                logger.info("OpenAI API connection successful (new client)")
            elif self.image_generator.openai_version == "legacy":
                # Legacy client test
                result = self.image_generator.client.Model.list()
                results["openai"] = True
                results["openai_details"] = "Models list retrieved"
                logger.info("OpenAI API connection successful (legacy client)")
            else:
                results["openai"] = False
                results["openai_details"] = "Unknown client version"
                logger.warning(f"Unknown OpenAI client version: {self.image_generator.openai_version}")
        except Exception as openai_err:
            logger.error(f"OpenAI API verification failed: {str(openai_err)}")
            results["openai_error"] = str(openai_err)
        
        # Overall status
        results["overall"] = results["whatsapp"] and results["openai"]
        
        return results

# Initialize marketing bot with enhanced error handling
logger.info("Initializing marketing bot with environment variables")
try:
    marketing_bot = MarketingBot(
        openai_key=OPENAI_API_KEY,
        waapi_token=WAAPI_API_TOKEN,
        waapi_instance_id=WAAPI_INSTANCE_ID
    )
    logger.info("Marketing bot initialized successfully")
    
    # Verify connections on startup
    connection_status = marketing_bot.verify_api_connections()
    if connection_status["overall"]:
        logger.info("All API connections verified successfully")
    else:
        logger.warning("Some API connections failed verification!")
        logger.warning(f"Connection status: {json.dumps(connection_status)}")
except Exception as init_error:
    logger.critical(f"Failed to initialize marketing bot: {str(init_error)}")
    logger.critical(traceback.format_exc())
    raise

###################
# SESSION MANAGEMENT
###################

def get_session(user_id: str, create_if_missing: bool = True) -> Dict:
    """Get or create a user session with thread safety"""
    with user_sessions_lock:
        if user_id in user_sessions:
            session = user_sessions[user_id]
            logger.debug(f"Retrieved existing session for {user_id}, state: {session.get('state', 'unknown')}")
            return session
        elif create_if_missing:
            logger.info(f"Creating new session for {user_id}")
            session = {
                "state": "waiting_for_command",
                "product_image": None,
                "details": {
                    "company_name": None,
                    "product_name": None,
                    "price": None,
                    "tagline": None,
                    "address": None
                },
                "last_message_time": datetime.now().timestamp(),
                "created_at": datetime.now().timestamp(),
                "message_count": 0,
                "history": [],
                "error_count": 0,
                "retry_count": 0,
                "last_error": None
            }
            user_sessions[user_id] = session
            return session
        else:
            logger.debug(f"No session exists for {user_id} and not creating new")
            return None

def update_session(user_id: str, updates: Dict) -> Dict:
    """Update a user session with thread safety and state validation"""
    with user_sessions_lock:
        if user_id not in user_sessions:
            logger.warning(f"Attempted to update non-existent session for {user_id}")
            return None
            
        session = user_sessions[user_id]
        
        # Validate state transitions
        if 'state' in updates:
            current_state = session.get('state', 'waiting_for_command')
            new_state = updates['state']
            
            valid_transitions = {
                'waiting_for_command': ['waiting_for_image', 'waiting_for_details'],
                'waiting_for_image': ['waiting_for_details', 'waiting_for_command'],
                'waiting_for_details': ['generating_image', 'waiting_for_command', 'waiting_for_image'],
                'generating_image': ['delivering_image', 'waiting_for_command'],
                'delivering_image': ['waiting_for_command']
            }
            
            # Special case: 'edit' command can reset to waiting_for_image from any state
            if new_state == 'waiting_for_image' and updates.get('command') == 'edit':
                logger.info(f"Resetting session to waiting_for_image due to edit command")
                session['state'] = 'waiting_for_image'
                session['product_image'] = None
                session['details'] = {}
                return session
            
            if current_state in valid_transitions and new_state not in valid_transitions[current_state]:
                logger.warning(f"Invalid state transition from {current_state} to {new_state}")
                return None
        
        # Update fields
        for key, value in updates.items():
            session[key] = value
        
        # Always update last_message_time
        session['last_message_time'] = datetime.now().timestamp()
        # Increment message count
        session['message_count'] = session.get('message_count', 0) + 1
        
        # Log state change if applicable
        if 'state' in updates:
            logger.info(f"Session state changed for {user_id}: {current_state} -> {new_state}")
        
        logger.debug(f"Updated session for {user_id}, new state: {session.get('state', 'unknown')}")
        return session

def cleanup_old_sessions():
    """Clean up old sessions that haven't been used recently"""
    current_time = datetime.now().timestamp()
    session_timeout = 24 * 60 * 60  # 24 hours in seconds
    
    with user_sessions_lock:
        expired_sessions = []
        for user_id, session in list(user_sessions.items()):
            last_activity = session.get('last_message_time', 0)
            if current_time - last_activity > session_timeout:
                expired_sessions.append(user_id)
                del user_sessions[user_id]
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)

###################
# MESSAGE HANDLERS
###################

def handle_text_message(from_number: str, text: str):
    """Handle incoming text messages with comprehensive error handling"""
    message_id = f"{from_number}_{int(time.time())}"
    
    try:
        # Clean and validate input
        text = text.strip() if text else ""
        logger.info(f"[{message_id}] Processing text message: '{text}' from {from_number}")
        
        if not text:
            logger.warning(f"[{message_id}] Empty message received from {from_number}")
            marketing_bot.waapi_client.send_message(
                from_number,
                "I received an empty message. To start creating a marketing image, please send 'edit'."
            )
            return
        
        # Get or create user session
        session = get_session(from_number)
        
        # Add to message history (limited to last 10 messages)
        history_entry = {"time": datetime.now().timestamp(), "type": "text", "content": text}
        session["history"] = (session.get("history", []) + [history_entry])[-10:]
        
        # Process different commands
        text_lower = text.lower()
        
        # Handle 'edit' command
        if text_lower == 'edit':
            logger.info(f"[{message_id}] User {from_number} sent 'edit' command")
            update_session(from_number, {
                'state': 'waiting_for_image',
                'product_image': None,
                'details': {}
            })
            
            marketing_bot.waapi_client.send_message(
                from_number,
                "Welcome to Marketing Image Editor! 📸\n\n"
                "Please send your product image to begin.\n\n"
                "After sending the image, I'll ask for details like company name, product name, price, etc."
            )
            logger.info(f"[{message_id}] Sent welcome message to {from_number}")
            return
        
        # Handle 'help' command
        if text_lower == 'help':
            logger.info(f"[{message_id}] User {from_number} requested help")
            
            help_message = (
                "📖 *Marketing Image Generator Help*\n\n"
                "*Available Commands:*\n"
                "• *edit* - Start creating a new marketing image\n"
                "• *generate* - Generate the image after providing details\n"
                "• *help* - Show this help message\n"
                "• *status* - Check your current session status\n"
                "• *reset* - Reset your session\n\n"
                "*How to use:*\n"
                "1. Send 'edit' to start\n"
                "2. Send a product image\n"
                "3. Provide product details as requested\n"
                "4. Send 'generate' to create your image\n\n"
                "Need more help? Contact support."
            )
            
            marketing_bot.waapi_client.send_message(from_number, help_message)
            return
        
        # Handle 'status' command
        if text_lower == 'status':
            logger.info(f"[{message_id}] User {from_number} requested status")
            
            status_message = "📊 *Session Status*\n\n"
            status_message += f"Current state: {session.get('state', 'Not started')}\n"
            
            if session.get('product_image'):
                status_message += "✅ Product image: Received\n"
            else:
                status_message += "❌ Product image: Not received\n"
            
            # Details summary
            details = session.get('details', {})
            status_message += "\n*Product Details:*\n"
            status_message += f"• Company: {details.get('company_name', '❌ Not provided')}\n"
            status_message += f"• Product: {details.get('product_name', '❌ Not provided')}\n"
            status_message += f"• Price: {details.get('price', '❌ Not provided')}\n"
            status_message += f"• Tagline: {details.get('tagline', '➖ Optional')}\n"
            status_message += f"• Address: {details.get('address', '➖ Optional')}\n\n"
            
            if session.get('state') == 'waiting_for_command':
                status_message += "To start a new image, send 'edit'."
            elif session.get('state') == 'waiting_for_image':
                status_message += "I'm waiting for you to send a product image."
            elif session.get('state') == 'waiting_for_details':
                if all([details.get('company_name'), details.get('product_name'), details.get('price')]):
                    status_message += "Ready to generate! Send 'generate' to create your marketing image."
                else:
                    status_message += "Please provide the remaining required details."
            
            marketing_bot.waapi_client.send_message(from_number, status_message)
            return
        
        # Handle 'reset' command
        if text_lower == 'reset':
            logger.info(f"[{message_id}] User {from_number} is resetting their session")
            update_session(from_number, {
                'state': 'waiting_for_command',
                'product_image': None,
                'details': {}
            })
            
            marketing_bot.waapi_client.send_message(
                from_number,
                "✅ Your session has been reset.\n\n"
                "To start creating a marketing image, send 'edit'."
            )
            return
        
        # Handle 'generate' command
        if text_lower == 'generate':
            logger.info(f"[{message_id}] User {from_number} sent 'generate' command")
            
            # Verify we're in the right state and have all required info
            if session.get('state') != 'waiting_for_details':
                logger.warning(f"[{message_id}] User tried to generate in wrong state: {session.get('state')}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "You're not ready to generate an image yet.\n\n" +
                    ("Please send 'edit' to start the process." if session.get('state') == 'waiting_for_command' else
                     "Please send a product image first." if session.get('state') == 'waiting_for_image' else
                     "Please complete providing product details first.")
                )
                return
            
            # Check for product image
            if not session.get('product_image') or not os.path.exists(session.get('product_image')):
                logger.warning(f"[{message_id}] User tried to generate without valid image")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "❌ Product image not found or invalid.\n\n"
                    "Please start over by sending 'edit'."
                )
                return
            
            # Check for required details
            details = session.get('details', {})
            required_fields = ['company_name', 'product_name', 'price']
            missing = [field.replace('_', ' ').title() for field in required_fields if not details.get(field)]
            
            if missing:
                logger.warning(f"[{message_id}] User tried to generate with missing details: {missing}")
                missing_text = ', '.join(missing)
                marketing_bot.waapi_client.send_message(
                    from_number,
                    f"❌ Missing required details: {missing_text}\n\n"
                    "Please provide all required information before generating."
                )
                return
            
            # All checks passed - now generate the image
            logger.info(f"[{message_id}] All validation passed, sending generation message")
            
            # Send generation in progress message
            sent = marketing_bot.waapi_client.send_message(
                from_number,
                "✨ Generating your marketing image...\n\n"
                "This may take up to 30 seconds. Please wait."
            )
            
            if not sent.get('success'):
                logger.error(f"[{message_id}] Failed to send generation message: {sent.get('error')}")
                # Continue anyway - the generation might still work
            
            # Determine product type from name if possible
            product_name = details.get('product_name', '').lower()
            product_type = "product"  # Default
            
            if any(term in product_name for term in ['coffee', 'tea', 'drink', 'juice', 'smoothie', 'latte']):
                product_type = "beverage"
            elif any(term in product_name for term in ['food', 'meal', 'dish', 'burger', 'sandwich', 'pizza']):
                product_type = "food"
            
            logger.info(f"[{message_id}] Determined product type: {product_type}")
            
            # Process the request
            logger.info(f"[{message_id}] Starting image generation process")
            result = marketing_bot.process_request(
                from_number,
                session.get('product_image'),
                details,
                product_type
            )
            
            if result.get('success'):
                logger.info(f"[{message_id}] Image generated successfully: {result.get('image_path')}")
                
                # Get image path
                image_path = result.get('image_path')
                
                # Read the image for base64 encoding
                logger.info(f"[{message_id}] Reading image file for base64 encoding")
                try:
                    with open(image_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                    
                    # Send the generated image
                    logger.info(f"[{message_id}] Sending generated image to {from_number}")
                    media_result = marketing_bot.waapi_client.send_media(
                        from_number,
                        "🎉 Here's your marketing image!\n\n"
                        "To create another image, send 'edit' again.",
                        media_base64=img_base64,
                        filename=os.path.basename(image_path)
                    )
                    
                    if media_result.get('success'):
                        logger.info(f"[{message_id}] Image sent successfully")
                    else:
                        logger.error(f"[{message_id}] Failed to send image: {media_result.get('error')}")
                        
                        # Try alternate approach - send image URL if available
                        if result.get('image_url'):
                            logger.info(f"[{message_id}] Trying to send image URL as fallback")
                            marketing_bot.waapi_client.send_message(
                                from_number,
                                f"I created your marketing image but couldn't send it directly.\n\n"
                                f"You can view and download it here:\n{result.get('image_url')}\n\n"
                                f"To create another image, send 'edit' again."
                            )
                        else:
                            # Last resort - just send error
                            marketing_bot.waapi_client.send_message(
                                from_number,
                                "I created your marketing image but couldn't send it. Please try again by sending 'edit'."
                            )
                    
                except Exception as file_error:
                    logger.error(f"[{message_id}] Error reading or sending image file: {str(file_error)}")
                    logger.error(traceback.format_exc())
                    
                    # Try to send image URL as fallback
                    if result.get('image_url'):
                        marketing_bot.waapi_client.send_message(
                            from_number,
                            f"I created your marketing image but encountered an error sending it.\n\n"
                            f"You can view and download it here:\n{result.get('image_url')}\n\n"
                            f"To create another image, send 'edit' again."
                        )
                    else:
                        marketing_bot.waapi_client.send_message(
                            from_number,
                            "I created your marketing image but encountered an error. Please try again by sending 'edit'."
                        )
                
                # Reset state regardless of send success
                update_session(from_number, {'state': 'waiting_for_command'})
                logger.info(f"[{message_id}] Session state reset to 'waiting_for_command'")
            else:
                logger.error(f"[{message_id}] Failed to generate image: {result.get('error')}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "❌ Sorry, I couldn't generate the image due to a technical issue.\n\n"
                    "Please try again or send 'edit' to start over."
                )
            return
        
        # Handle conversation flow based on state
        if session['state'] == 'waiting_for_image':
            logger.info(f"[{message_id}] User sent text while waiting for image")
            marketing_bot.waapi_client.send_message(
                from_number,
                "I'm waiting for you to send a product image.\n\n"
                "Please send an image file, not text."
            )
            return
        
        elif session['state'] == 'waiting_for_details':
            logger.info(f"[{message_id}] User {from_number} sent details: {text}")
            
            # Parse the details
            detail_provided = False
            details = session.get('details', {})
            
            # Try structured format first (key: value pairs)
            lines = text.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if 'company' in key:
                        details['company_name'] = value
                        detail_provided = True
                        logger.info(f"[{message_id}] Set company_name: {value}")
                    elif 'product' in key:
                        details['product_name'] = value
                        detail_provided = True
                        logger.info(f"[{message_id}] Set product_name: {value}")
                    elif 'price' in key:
                        details['price'] = value
                        detail_provided = True
                        logger.info(f"[{message_id}] Set price: {value}")
                    elif 'tagline' in key:
                        details['tagline'] = value
                        detail_provided = True
                        logger.info(f"[{message_id}] Set tagline: {value}")
                    elif 'address' in key or 'location' in key:
                        details['address'] = value
                        detail_provided = True
                        logger.info(f"[{message_id}] Set address: {value}")
            
            # If no structured details, interpret as single value in sequence
            if not detail_provided:
                logger.info(f"[{message_id}] No structured details found, interpreting as single value")
                
                if not details.get('company_name'):
                    details['company_name'] = text
                    logger.info(f"[{message_id}] Set company_name: {text}")
                elif not details.get('product_name'):
                    details['product_name'] = text
                    logger.info(f"[{message_id}] Set product_name: {text}")
                elif not details.get('price'):
                    details['price'] = text
                    logger.info(f"[{message_id}] Set price: {text}")
                elif not details.get('tagline'):
                    details['tagline'] = text
                    logger.info(f"[{message_id}] Set tagline: {text}")
                elif not details.get('address'):
                    details['address'] = text
                    logger.info(f"[{message_id}] Set address: {text}")
                else:
                    # All fields already filled, update the last field (address)
                    details['address'] = text
                    logger.info(f"[{message_id}] Updated address: {text}")
            
            # Update session with new details
            update_session(from_number, {'details': details})
            
            # Send updated status and next step
            logger.info(f"[{message_id}] Sending status update to {from_number}")
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
            
            marketing_bot.waapi_client.send_message(from_number, status_msg)
            return
        
        # Default state - waiting for command
        else:
            logger.info(f"[{message_id}] User {from_number} sent message in default state")
            marketing_bot.waapi_client.send_message(
                from_number,
                "👋 Welcome to Marketing Image Generator!\n\n"
                "To create a marketing image, send 'edit' to start.\n"
                "For help, send 'help'."
            )
            
    except Exception as e:
        logger.error(f"[{message_id}] Error handling text message: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, an error occurred while processing your message. Please try again.\n"
                "Send 'edit' to start over or 'help' for assistance."
            )
        except Exception as send_error:
            logger.error(f"[{message_id}] Failed to send error message: {str(send_error)}")

def handle_image_message(from_number: str, media_data):
    """Handle incoming image messages with robust error handling"""
    message_id = f"{from_number}_{int(time.time())}"
    
    try:
        logger.info(f"[{message_id}] Image received from {from_number}. Processing...")
        
        # Get or create user session
        session = get_session(from_number)
        
        # Add to message history
        history_entry = {"time": datetime.now().timestamp(), "type": "image"}
        session["history"] = (session.get("history", []) + [history_entry])[-10:]
        
        # Debug session state
        logger.info(f"[{message_id}] Current session state: {session.get('state')}")
        
        # Check if we're in the right state to receive an image
        if session.get('state') != 'waiting_for_image':
            logger.warning(f"[{message_id}] Received image but session state is {session.get('state')}, not waiting_for_image")
            marketing_bot.waapi_client.send_message(
                from_number,
                "I wasn't expecting an image right now.\n\n" +
                ("To start the process, please send 'edit' first." if session.get('state') == 'waiting_for_command' else
                 "I already have your image. Please provide the product details." if session.get('state') == 'waiting_for_details' else
                 "Please send 'edit' to start over.")
            )
            return
        
        # Process the image
        try:
            # Try to extract image data in various formats
            image_bytes = None
            
            if media_data:
                # Try various media data formats
                if isinstance(media_data, dict):
                    if 'data' in media_data and media_data['data']:
                        logger.info(f"[{message_id}] Found base64 data in media_data['data']")
                        try:
                            image_bytes = base64.b64decode(media_data['data'])
                        except Exception as b64_err:
                            logger.error(f"[{message_id}] Base64 decode error: {str(b64_err)}")
                    
                    elif 'base64' in media_data and media_data['base64']:
                        logger.info(f"[{message_id}] Found base64 data in media_data['base64']")
                        try:
                            image_bytes = base64.b64decode(media_data['base64'])
                        except Exception as b64_err:
                            logger.error(f"[{message_id}] Base64 decode error: {str(b64_err)}")
                            
                    elif 'url' in media_data and media_data['url']:
                        logger.info(f"[{message_id}] Found URL in media_data['url']")
                        try:
                            response = requests.get(media_data['url'], timeout=30)
                            if response.status_code == 200:
                                image_bytes = response.content
                            else:
                                logger.error(f"[{message_id}] Failed to download image: HTTP {response.status_code}")
                        except Exception as url_err:
                            logger.error(f"[{message_id}] URL download error: {str(url_err)}")
                
                elif isinstance(media_data, str) and (len(media_data) > 100):
                    # Try to interpret as direct base64 string
                    logger.info(f"[{message_id}] Attempting to parse media_data as direct base64 string")
                    try:
                        image_bytes = base64.b64decode(media_data)
                    except Exception as direct_b64_err:
                        logger.error(f"[{message_id}] Direct base64 decode error: {str(direct_b64_err)}")
            
            # If we couldn't extract image data, use placeholder
            if not image_bytes:
                logger.warning(f"[{message_id}] Couldn't extract image data, using placeholder")
                
                # Create placeholder image
                img_buffer = ImageProcessor.create_placeholder_image(512, 512)
                image_bytes = img_buffer.getvalue()
                
                # Inform the user
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "⚠️ I received your image but couldn't access its data due to API limitations.\n\n"
                    "I'll continue with a placeholder image for testing. In production, your actual image would be used."
                )
            
            # Validate image data before saving
            try:
                img = Image.open(BytesIO(image_bytes))
                width, height = img.size
                format = img.format
                logger.info(f"[{message_id}] Image validated: {width}x{height} {format}")
            except Exception as validate_err:
                logger.error(f"[{message_id}] Invalid image data: {str(validate_err)}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "❌ The image you sent appears to be invalid or corrupted.\n\n"
                    "Please try sending a different image."
                )
                return
            
            # Save the image
            timestamp = generate_timestamp()
            safe_number = from_number.replace('@c.us', '').replace('+', '')
            filename = f"whatsapp_image_{safe_number}_{timestamp}.jpg"
            image_path = os.path.join("images/input", filename)
            
            logger.info(f"[{message_id}] Saving image to {image_path}")
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Verify the file was saved correctly
            if not os.path.exists(image_path):
                logger.error(f"[{message_id}] Failed to save image at {image_path}")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "❌ There was an error saving your image. Please try again."
                )
                return
                
            file_size = os.path.getsize(image_path)
            if file_size < 100:  # Extremely small file, likely corrupt
                logger.error(f"[{message_id}] Saved image is too small ({file_size} bytes), likely corrupt")
                marketing_bot.waapi_client.send_message(
                    from_number,
                    "❌ The saved image appears to be invalid. Please try sending a different image."
                )
                return
            
            logger.info(f"[{message_id}] Image saved to {image_path} ({file_size} bytes)")
            
            # Update session with image path and change state
            update_session(from_number, {
                'product_image': image_path,
                'state': 'waiting_for_details'
            })
            
            # Send confirmation and request details
            marketing_bot.waapi_client.send_message(
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
                "When you're ready to generate the image, send 'generate'"
            )
            logger.info(f"[{message_id}] Successfully processed image and sent response")
            
        except Exception as e:
            logger.error(f"[{message_id}] Error processing image: {str(e)}")
            logger.error(traceback.format_exc())
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, I couldn't process your image. Please try again."
            )
            
    except Exception as e:
        logger.error(f"[{message_id}] Error handling image message: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            marketing_bot.waapi_client.send_message(
                from_number,
                "Sorry, I couldn't process your image. Please try again.\n"
                "Start over by sending 'edit'."
            )
        except Exception as send_error:
            logger.error(f"[{message_id}] Failed to send error message: {str(send_error)}")

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
        "version": "1.2.1",
        "documentation": "/docs",
        "health": "/health",
        "test": "/test-openai"
    })

@app.route('/docs')
def docs():
    """Show API documentation"""
    return jsonify({
        "info": {
            "title": "Marketing Image Generator API",
            "version": "1.2.1",
            "description": "WhatsApp-based marketing image generation service"
        },
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Service information"},
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/test-openai", "method": "GET", "description": "Test OpenAI connection"},
            {"path": "/webhook", "method": "POST", "description": "WaAPI webhook endpoint"},
            {"path": "/images/<path>", "method": "GET", "description": "Serve generated images"},
            {"path": "/stats", "method": "GET", "description": "Service statistics (admin)"},
            {"path": "/sessions", "method": "GET", "description": "Active sessions (admin)"}
        ]
    })

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    try:
        # Basic health metrics
        stats = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.2.1",
            "sessions": len(user_sessions),
            "memory_usage": {
                "sessions_kb": len(json.dumps(user_sessions)) // 1024
            }
        }
        
        # Check if directories are accessible
        dirs_ok = all(os.path.isdir(d) for d in ['images/input', 'images/output', 'logs'])
        stats["directories_ok"] = dirs_ok
        
        # Check API connections if available
        try:
            stats["api_status"] = marketing_bot.verify_api_connections()
        except:
            stats["api_status"] = "unavailable"
        
        # Check processing times
        cleanup_old_sessions()
        stats["sessions_after_cleanup"] = len(user_sessions)
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/test-openai')
def test_openai():
    """Test the OpenAI connection"""
    try:
        logger.info("Testing OpenAI connection")
        start_time = time.time()
        
        # Initial connection info
        result = {
            "timestamp": datetime.now().isoformat(),
            "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
            "tests": {}
        }
        
        # Try model list
        try:
            models = openai_client.models.list()
            result["tests"]["models_list"] = {
                "success": True,
                "model_count": len(models.data) if hasattr(models, 'data') else "unknown"
            }
        except Exception as model_err:
            result["tests"]["models_list"] = {
                "success": False,
                "error": str(model_err)
            }
        
        # Try a simple DALL-E 2 generation (faster/cheaper than DALL-E 3)
        try:
            logger.info("Testing with OpenAI client - DALL-E 2")
            image_result = openai_client.images.generate(
                model="dall-e-2",
                prompt="A simple test image of a red cube",
                size="256x256",
                n=1
            )
            result["tests"]["image_generation"] = {
                "success": True,
                "model": "dall-e-2"
            }
        except Exception as img_err:
            result["tests"]["image_generation"] = {
                "success": False,
                "error": str(img_err)
            }
        
        # Overall status
        result["success"] = any(test.get("success", False) for test in result["tests"].values())
        result["duration_seconds"] = time.time() - start_time
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"OpenAI test failed: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })

@app.route('/stats')
def stats():
    """Admin statistics endpoint"""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "sessions": {
            "total": len(user_sessions),
            "states": {}
        },
        "messages": {
            "processed": len(processed_messages)
        },
        "storage": {
            "input_images": 0,
            "output_images": 0
        }
    }
    
    # Session state counts
    for session in user_sessions.values():
        state = session.get('state', 'unknown')
        stats["sessions"]["states"][state] = stats["sessions"]["states"].get(state, 0) + 1
    
    # Storage stats
    try:
        input_files = os.listdir('images/input')
        output_files = os.listdir('images/output')
        stats["storage"]["input_images"] = len(input_files)
        stats["storage"]["output_images"] = len(output_files)
        
        # Total sizes
        input_size = sum(os.path.getsize(os.path.join('images/input', f)) for f in input_files)
        output_size = sum(os.path.getsize(os.path.join('images/output', f)) for f in output_files)
        stats["storage"]["input_size_mb"] = round(input_size / (1024 * 1024), 2)
        stats["storage"]["output_size_mb"] = round(output_size / (1024 * 1024), 2)
    except Exception as dir_err:
        stats["storage"]["error"] = str(dir_err)
    
    return jsonify(stats)

@app.route('/sessions')
def get_sessions():
    """Admin endpoint to view sessions (for debugging)"""
    # Create a safe version of sessions (limited info)
    safe_sessions = {}
    
    for user_id, session in user_sessions.items():
        # Remove any sensitive data
        safe_session = {
            "state": session.get('state', 'unknown'),
            "has_image": bool(session.get('product_image')),
            "details_provided": len(session.get('details', {})),
            "last_activity": datetime.fromtimestamp(session.get('last_message_time', 0)).isoformat(),
            "created_at": datetime.fromtimestamp(session.get('created_at', 0)).isoformat(),
            "message_count": session.get('message_count', 0)
        }
        
        # Masked user ID (for privacy)
        masked_id = user_id[:4] + "****" + user_id[-4:] if len(user_id) > 8 else "****"
        safe_sessions[masked_id] = safe_session
    
    return jsonify({
        "count": len(safe_sessions),
        "sessions": safe_sessions
    })

@app.route('/images/<path:path>')
def serve_images(path):
    """Serve images from the images directory"""
    logger.debug(f"Serving image: {path}")
    
    # Security check - prevent directory traversal
    if '..' in path:
        logger.warning(f"Attempted directory traversal: {path}")
        return jsonify({"error": "Invalid path"}), 400
        
    directory, filename = os.path.split(path)
    
    # Validate the directory is within our expected structure
    if directory not in ['input', 'output']:
        logger.warning(f"Attempted to access invalid directory: {directory}")
        return jsonify({"error": "Invalid directory"}), 400
    
    # Check if file exists before serving
    full_path = os.path.join('images', directory, filename)
    if not os.path.exists(full_path):
        logger.warning(f"Attempted to access non-existent file: {full_path}")
        return jsonify({"error": "File not found"}), 404
    
    return send_from_directory(os.path.join('images', directory), filename)

@app.route('/clear-sessions')
def clear_sessions():
    """Admin endpoint to clear all sessions"""
    with user_sessions_lock:
        session_count = len(user_sessions)
        user_sessions.clear()
    logger.info(f"Cleared {session_count} user sessions")
    return jsonify({"status": "success", "cleared": session_count})

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming WhatsApp messages via WaAPI webhook"""
    webhook_id = f"webhook_{int(time.time())}_{hash(str(request.remote_addr))}"
    
    try:
        # Extract the webhook data
        if not request.is_json:
            logger.warning(f"[{webhook_id}] Received non-JSON webhook payload")
            return jsonify({"status": "error", "message": "Expected JSON payload"}), 400
            
        webhook_data = request.json
        
        # Log webhook with truncated/sanitized data
        truncated_data = str(webhook_data)[:1000] + "..." if len(str(webhook_data)) > 1000 else str(webhook_data)
        logger.info(f"[{webhook_id}] Received webhook: {truncated_data}")
        
        # Validate webhook structure
        if not isinstance(webhook_data, dict):
            logger.warning(f"[{webhook_id}] Invalid webhook data format")
            return jsonify({"status": "error", "message": "Invalid data format"}), 400
            
        # Check for webhook verification if configured
        if WEBHOOK_SECRET and request.args.get('verify') == 'true':
            secret = request.args.get('secret')
            if secret != WEBHOOK_SECRET:
                logger.warning(f"[{webhook_id}] Invalid webhook verification secret")
                return jsonify({"status": "error", "message": "Invalid verification secret"}), 403
            else:
                logger.info(f"[{webhook_id}] Webhook verification successful")
                return jsonify({"status": "success", "message": "Webhook verified"})
        
        # Extract message ID to prevent duplicate processing
        message_data = safe_get(webhook_data, 'data.message', {})
        
        # Debug the message structure at detailed level
        logger.debug(f"[{webhook_id}] Message data structure: {json.dumps(message_data)}")
        
        # Extract message ID carefully with multiple fallback options
        message_id = None
        
        # Try multiple locations for message ID
        if isinstance(message_data, dict):
            # Direct ID field
            if 'id' in message_data:
                msg_id = message_data['id']
                if isinstance(msg_id, dict) and '_serialized' in msg_id:
                    message_id = msg_id['_serialized']
                else:
                    message_id = str(msg_id)
            
            # ID in _data field
            elif '_data' in message_data and isinstance(message_data['_data'], dict):
                data_id = message_data['_data'].get('id')
                if isinstance(data_id, dict) and '_serialized' in data_id:
                    message_id = data_id['_serialized']
                elif data_id:
                    message_id = str(data_id)
        
        # If no ID found, generate one
        if not message_id:
            # Generate pseudo-ID from timestamp and content hash
            timestamp = datetime.now().timestamp()
            content_hash = hashlib.md5(str(message_data).encode()).hexdigest()[:8]
            message_id = f"gen_{timestamp}_{content_hash}"
            logger.warning(f"[{webhook_id}] No message ID found, generated: {message_id}")
        
        # Skip if we've already processed this message
        with processed_messages_lock:
            if message_id in processed_messages:
                logger.info(f"[{webhook_id}] Skipping duplicate message: {message_id}")
                return jsonify({"status": "success", "message": "Duplicate message skipped"})
            
            # Mark as processed
            processed_messages[message_id] = datetime.now().timestamp()
            
            # Limit cache size by removing old entries (keep last 100)
            if len(processed_messages) > 100:
                oldest = sorted(processed_messages.items(), key=lambda x: x[1])[0][0]
                processed_messages.pop(oldest)
                logger.debug(f"[{webhook_id}] Removed oldest message {oldest} from processed cache")
        
        # Determine event type
        event_type = webhook_data.get('event', 'unknown')
        
        # Check if this is a message event
        if event_type in ['message', 'message_create', 'chat']:
            # Extract message details with robust fallbacks
            message_type = ''
            from_number = ''
            body = ''
            
            # Try multiple paths to extract sender number
            possible_from_paths = [
                'data.message.from',
                'data.message._data.from',
                'data.from',
                'message.from',
                'sender'
            ]
            
            for path in possible_from_paths:
                from_number = safe_get(webhook_data, path, '')
                if from_number:
                    break
            
            # Try multiple paths to extract message type
            possible_type_paths = [
                'data.message.type',
                'data.message._data.type',
                'data.type',
                'message.type',
                'type'
            ]
            
            for path in possible_type_paths:
                message_type = safe_get(webhook_data, path, '')
                if message_type:
                    break
            
            # Try multiple paths to extract message body
            possible_body_paths = [
                'data.message.body',
                'data.message._data.body',
                'data.body',
                'message.body',
                'body'
            ]
            
            for path in possible_body_paths:
                raw_body = safe_get(webhook_data, path, '')
                if raw_body:
                    if isinstance(raw_body, str):
                        body = raw_body.strip()
                    break
            
            # Check for media in multiple ways
            has_media = False
            
            # Direct media flags
            media_flags = [
                'data.message.hasMedia',
                'data.message._data.hasMedia',
                'data.hasMedia',
                'message.hasMedia',
                'hasMedia'
            ]
            
            for flag_path in media_flags:
                if safe_get(webhook_data, flag_path, False):
                    has_media = True
                    break
            
            # Check media type
            if message_type in ['image', 'sticker', 'video', 'document']:
                has_media = True
            
            # Check for mediaData fields
            media_data_paths = [
                'data.message.mediaData',
                'data.message._data.mediaData',
                'data.media',
                'message.mediaData',
                'mediaData'
            ]
            
            for media_path in media_data_paths:
                if safe_get(webhook_data, media_path):
                    has_media = True
                    break
            
            logger.info(f"[{webhook_id}] Received message from {from_number}: {body[:50]}{'...' if len(body) > 50 else ''}, Media: {has_media}, Type: {message_type}")
            
            # Handle group messages differently
            if from_number and '@g.us' in from_number:
                logger.info(f"[{webhook_id}] Received group message from {from_number}")
                # Extract the actual sender from the group
                author = safe_get(webhook_data, 'data.message._data.author', '')
                if author and '@c.us' in author:
                    from_number = author
                    logger.info(f"[{webhook_id}] Extracted sender from group: {from_number}")
                else:
                    logger.warning(f"[{webhook_id}] Could not extract sender from group message")
                    return jsonify({"status": "success", "message": "Group message processed"})
            
            # Validate phone number format
            if not from_number or '@c.us' not in str(from_number):
                logger.error(f"[{webhook_id}] Invalid or missing phone number format: {from_number}")
                return jsonify({"status": "error", "message": "Invalid or missing phone number format"})
            
            # Extract media data with multiple fallbacks
            media_data = None
            
            # Try multiple paths to find media data
            media_data_paths = [
                'data.media',
                'data.message.mediaData',
                'data.message._data.mediaData',
                'message.mediaData',
                'mediaData'
            ]
            
            for media_path in media_data_paths:
                media_data = safe_get(webhook_data, media_path)
                if media_data:
                    logger.debug(f"[{webhook_id}] Found media data at {media_path}")
                    break
            
            # Handle media messages
            if has_media:
                logger.info(f"[{webhook_id}] Detected media in message from {from_number}, type: {message_type}")
                
                if not media_data:
                    logger.info(f"[{webhook_id}] Media indicated but no media data in webhook, will use placeholder")
                
                # Process as image
                handle_image_message(from_number, media_data)
                return jsonify({"status": "success", "message": "Media processed"})
                
            # Handle text messages
            handle_text_message(from_number, body)
            return jsonify({"status": "success", "message": "Text processed"})
        
        # Handle group join events
        elif event_type == 'group_join':
            logger.info(f"[{webhook_id}] Received group join event")
            return jsonify({"status": "success", "message": "Group join event processed"})
        
        # Handle other event types
        logger.info(f"[{webhook_id}] Received non-message event: {event_type}")
        return jsonify({"status": "success", "message": f"Event {event_type} acknowledged"})
    
    except Exception as e:
        logger.error(f"[{webhook_id}] Webhook error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

###################
# SESSION CLEANUP
###################

def start_cleanup_scheduler():
    """Start a background thread to periodically clean up old sessions"""
    def cleanup_task():
        while True:
            try:
                logger.debug("Running scheduled session cleanup")
                cleanup_count = cleanup_old_sessions()
                logger.debug(f"Cleaned up {cleanup_count} old sessions")
                
                # Also clean up processed messages cache
                with processed_messages_lock:
                    current_time = datetime.now().timestamp()
                    expired_threshold = current_time - (6 * 60 * 60)  # 6 hours
                    old_messages = [msg_id for msg_id, timestamp in processed_messages.items() if timestamp < expired_threshold]
                    
                    for msg_id in old_messages:
                        processed_messages.pop(msg_id)
                    
                    if old_messages:
                        logger.debug(f"Cleaned up {len(old_messages)} old processed message entries")
                
                # Run every hour
                time.sleep(60 * 60)
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                time.sleep(5 * 60)  # Wait 5 minutes on error
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()
    logger.info("Started session cleanup scheduler")
    return cleanup_thread

###################
# MAIN APPLICATION
###################

if __name__ == '__main__':
    # Start background cleanup task
    cleanup_thread = start_cleanup_scheduler()
    
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv('PORT', 8080))
    
    # Log startup information
    logger.info(f"Starting Marketing Bot API (WhatsApp Version) on port {port}")
    logger.info(f"OpenAI API client version: {OPENAI_VERSION}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=False)  # Set debug=False for production