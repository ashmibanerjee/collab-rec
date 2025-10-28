from jinja2 import Environment, FileSystemLoader
from src.data_directories import *
from num2words import num2words

ROLES = ["preference", "interest", "diversity"]


def render_prompt(round, role, template, specs={}):
    """
    Helper function to render prompts. Here, role refers to "sys" or "user", not the agent role. Note: Please pass filters as a dictionary. The key should match with the name of the template variable to account for error-free augmentation. 
    """
    # preprocess integer params to words
    context = specs.copy()
    # print(role, context)
    if context.get("k_offer"):
        context.update({"k_offer":num2words(specs["k_offer"])})
    
    if context.get("k_reject"):
        context.update({"k_reject":num2words(specs["k_reject"])})

    if context.get("k_invalid"):
        context.update({"k_invalid":num2words(specs["k_invalid"])})

    if "init" in round: 
        env = Environment(loader=FileSystemLoader(f"{prompts_dir}/agents/init/"))
    elif "iter" in round: 
        env = Environment(loader=FileSystemLoader(f"{prompts_dir}/agents/iter/"))
    elif "feedback" in round:
        env = Environment(loader=FileSystemLoader(f"{prompts_dir}/agents/feedback/"))
    elif "single" in round: 
        env = Environment(loader=FileSystemLoader(f"{prompts_dir}/agents/single-agent/"))
    else: 
        print("Error, unable to parse prompt!")
        return None
    
    template = env.get_template(template)

    if len(context):
        prompt = {
            "role": role,
            "content": template.render(context)
        }
    else: 
       prompt = {
            "role": role,
            "content": template.render()
        } 

    return prompt
