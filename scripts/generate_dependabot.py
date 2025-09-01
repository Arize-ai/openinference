import os
import yaml

def generate_dependabot_config():
    config = {"version": 2, "updates": []}
    
    # Add your base packages
    base_configs = [
        {"package-ecosystem": "gradle", "directory": "/java"},
        {"package-ecosystem": "npm", "directory": "/js"},
        {"package-ecosystem": "pip", "directory": "/python"}
    ]
    
    # Find all instrumentation packages
    py_instrumentation_dir = "python/instrumentation" 
    for package in os.listdir(py_instrumentation_dir):
        if package.startswith("openinference-instrumentation-"):
            path = f"{py_instrumentation_dir}/{package}"
            if os.path.exists(f"{path}/pyproject.toml"):
                base_configs.append({
                    "package-ecosystem": "pip",
                    "directory": path
                })
    npm_instrumentation_dir = "js/packages" 
    for package in os.listdir(npm_instrumentation_dir):
        if package.startswith("openinference-"):
            path = f"{npm_instrumentation_dir}/{package}" 
            if os.path.exists(f"{path}/package.json"):
                base_configs.append({
                    "package-ecosystem": "npm",
                    "directory": path
                })
    # Add common settings to all
    for cfg in base_configs:
        cfg["schedule"] = {"interval": "weekly"}
        cfg["ignore"] = [{"dependency-name": "*", "update-types": ["version-update:semver-major", "version-update:semver-minor", "version-update:semver-patch"]}]
        config["updates"].append(cfg)
    
    return yaml.dump(config, default_flow_style=False)

print(generate_dependabot_config())
