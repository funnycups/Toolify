import os
import yaml
from typing import List, Dict, Any, Set
from pydantic import BaseModel, Field, validator


class ServerConfig(BaseModel):
    """Server configuration"""
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    host: str = Field(default="0.0.0.0", description="Server host address")
    timeout: int = Field(default=180, ge=1, description="Request timeout (seconds)")


class UpstreamService(BaseModel):
    """Upstream service configuration"""
    name: str = Field(description="Service name")
    base_url: str = Field(description="Service base URL")
    api_key: str = Field(description="API key")
    models: List[str] = Field(description="List of supported models")
    description: str = Field(default="", description="Service description")
    is_default: bool = Field(default=False, description="Is default service")
    
    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')
    
    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError('api_key cannot be empty')
        return v
    
    @validator('models')
    def validate_models(cls, v):
        if not v or len(v) == 0:
            raise ValueError('models list cannot be empty')
        for model in v:
            if not model or model.strip() == "":
                raise ValueError('model name cannot be empty')
        return v


class ClientAuthConfig(BaseModel):
    """Client authentication configuration"""
    allowed_keys: List[str] = Field(description="List of allowed client API keys")
    
    @validator('allowed_keys')
    def validate_allowed_keys(cls, v):
        if not v or len(v) == 0:
            raise ValueError('allowed_keys cannot be empty')
        for key in v:
            if not key or key.strip() == "":
                raise ValueError('API key cannot be empty')
        return v


class FeaturesConfig(BaseModel):
    """Feature configuration"""
    enable_function_calling: bool = Field(default=True, description="Enable function calling")
    enable_logging: bool = Field(default=True, description="Enable logging")
    convert_developer_to_system: bool = Field(default=True, description="Convert developer role to system role")


class AppConfig(BaseModel):
    """Application full configuration"""
    server: ServerConfig = Field(default_factory=ServerConfig)
    upstream_services: List[UpstreamService] = Field(description="List of upstream services")
    client_authentication: ClientAuthConfig = Field(description="Client authentication configuration")
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    
    @validator('upstream_services')
    def validate_upstream_services(cls, v):
        if not v or len(v) == 0:
            raise ValueError('upstream_services cannot be empty')
        
        default_services = [service for service in v if service.is_default]
        if len(default_services) == 0:
            raise ValueError('Must have at least one default upstream service (is_default: true)')
        if len(default_services) > 1:
            raise ValueError('Only one upstream service can be marked as default')
        
        all_models = []
        for service in v:
            all_models.extend(service.models)
        
        duplicates = []
        seen = set()
        for model in all_models:
            if model in seen:
                duplicates.append(model)
            seen.add(model)
        
        if duplicates:
            raise ValueError(f'Duplicate models found: {duplicates}')
        
        return v


class ConfigLoader:
    """Configuration loader"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config: AppConfig = None
    
    def load_config(self) -> AppConfig:
        """Load configuration file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file '{self.config_path}' not found. "
                f"Please copy 'config.example.yaml' to '{self.config_path}' and modify the configuration as needed."
            )
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Configuration file format error: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read configuration file: {e}")
        
        if not config_data:
            raise ValueError("Configuration file is empty")
        
        try:
            self._config = AppConfig(**config_data)
            return self._config
        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")
    
    @property
    def config(self) -> AppConfig:
        """Get configuration object"""
        if self._config is None:
            self.load_config()
        return self._config
    
    def get_model_to_service_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Get model to service mapping"""
        config = self.config
        model_mapping = {}
        
        for service in config.upstream_services:
            service_info = {
                "name": service.name,
                "base_url": service.base_url,
                "api_key": service.api_key,
                "description": service.description,
                "is_default": service.is_default
            }
            
            for model in service.models:
                model_mapping[model] = service_info
        
        return model_mapping
    
    def get_default_service(self) -> Dict[str, Any]:
        """Get default service configuration"""
        config = self.config
        for service in config.upstream_services:
            if service.is_default:
                return {
                    "name": service.name,
                    "base_url": service.base_url,
                    "api_key": service.api_key,
                    "description": service.description,
                    "is_default": service.is_default
                }
        raise ValueError("No default service configured")
    
    def get_allowed_client_keys(self) -> Set[str]:
        """Get set of allowed client keys"""
        return set(self.config.client_authentication.allowed_keys)
    
    def is_logging_enabled(self) -> bool:
        """Check if logging is enabled"""
        return self.config.features.enable_logging
    
    def get_features_config(self) -> Dict[str, bool]:
        """Get feature configuration"""
        return {
            "function_calling": self.config.features.enable_function_calling,
            "logging": self.config.features.enable_logging,
            "convert_developer_to_system": self.config.features.convert_developer_to_system
        }


config_loader = ConfigLoader()