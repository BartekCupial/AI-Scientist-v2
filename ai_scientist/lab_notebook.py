import re
import argparse
import argparse
import json
import traceback
import os
import os.path as osp
import shutil
from enum import Enum

from ai_scientist.llm import (
    get_response_from_llm,
    create_client,
    AVAILABLE_LLMS,
)

from ai_scientist.perform_vlm_review import generate_vlm_img_review
from ai_scientist.vlm import create_client as create_vlm_client


notebook_system_message_template = """You are a ambitious AI researcher trying to create a lab notebook 
summarizing your experiments. Your supervisor will use this notebook to understand your experimental 
work, track progress, assess results, give feedback, and potentially guide future research directions. 
The notebook must be clear, experiments should be explicit, and enable the supervisor to understand everything 
without overloading him with too much information.

Your lab notebook should follow a structured format that enables quick comprehension and easy navigation:

## **General Guidelines**:
- Use clear, concise language that balances technical precision with readability
- Maintain consistent formatting and organization throughout
- Highlight key insights, breakthroughs, and decision points

## **Structure for Each Experiment Entry**:

- **Experiment Title**:
  - Clear, descriptive title that captures the experiment's purpose

- **Motivation & Hypothesis**:
  - Briefly explain why this experiment was conducted
  - State clear hypotheses or research questions being tested
  - Connect to broader research goals and previous experiments
  
- **Experimental Setup**:
  - Describe the methodology concisely but completely
  - Include key parameters, datasets, models, and configurations
  - Note any changes from previous experiments
  - Specify evaluation metrics and success criteria
  
- **Results**:
  - Present findings objectively with appropriate visualizations
  - Include quantitative results (tables, metrics) and qualitative observations
  - Report both positive and negative results honestly
  - Highlight unexpected findings or anomalies
  
- **Analysis & Interpretation**:
  - Interpret results in context of the original hypothesis
  - Compare with baseline methods or previous experiments
  - Identify potential causes for unexpected outcomes
  - Assess statistical significance and practical importance

- **Key Insights & Lessons Learned**:
  - Summarize the main takeaways from the experiment
  - Note methodological insights or technical discoveries
  - Document what worked well and what didn't
  - Identify potential improvements or alternative approaches

- **Next Steps**:
  - Outline immediate follow-up experiments or investigations
  - Suggest modifications based on current results
  - Note questions that emerged from this work
  
## **Best Practices**:
- Use clear headings and consistent formatting for easy scanning
- Include relevant plots, tables, and figures with proper captions
- Cross-reference related experiments and build narrative connections
- Balance detail with brevity - include essential information without overwhelming
- Use bullet points and numbered lists for clarity
- Maintain a running glossary of key terms and abbreviations
- Regular backup and version control of notebook entries

Remember: Your supervisor should be able to quickly understand your progress, assess the quality of your work, and provide meaningful guidance based on your documentation. The notebook should tell a coherent story of your research journey while serving as a reliable reference for future work.
"""

notebook_prompt = """Your primary goal is to create or update a lab notebook entry. This notebook is a critical 
tool for your supervisor to understand your experimental work, track progress, assess results, give feedback, and 
guide future research directions.

It is essential that the notebook entries are clear, concise, well-structured, and accurately reflect the experimental data and your analysis.

Please meticulously follow the structure and guidelines for each experiment entry.

The current task is to document experiments in the context of this idea:
```markdown
{idea_text}
```

We have the following experiment summaries (JSON):
```json
{summaries}
```

We also have a script used to produce the final plots (use this to see how the plots are generated and what names are used in the legend):
```python
{aggregator_code}
```
Please also consider which plots should naturally be grouped together as subfigures.

Available plots for the writeup (use these filenames):
```
{plot_list}
```

We also have VLM-based figure descriptions:
```
{plot_descriptions}
```

Your current Lab Notebook content (if this is an update to an existing notebook, this will be empty):
```markdown
{current_notebook}
```

Return the entire lab notebook content in Markdown format, enclosed in triple backticks with `markdown` syntax highlighting, as shown below:

```markdown
<UPDATED MARKDOWN LAB NOTEBOOK CONTENT>
```
"""



def create_lab_notebook(
    base_folder: str, 
    model: str = "gpt-4o-2024-05-13", 
    n_notebook_reflections: int = 3
) -> bool:
    """
    Create a lab notebook for the AI Scientist project.
    """
    notebook_folder = osp.join(base_folder, "notebook")
    figures_dir = osp.join(base_folder, "figures")
    
    # Cleanup any previous notebook folder and figures
    if osp.exists(notebook_folder):
        shutil.rmtree(notebook_folder)
    
    os.makedirs(notebook_folder, exist_ok=True)

    new_figures_dir = osp.join(notebook_folder, "figures")
    shutil.copytree(figures_dir, new_figures_dir)

    try:
        # Load idea text
        idea_text = ""
        research_idea_path = osp.join(base_folder, "research_idea.md")
        if osp.exists(research_idea_path):
            with open(research_idea_path, "r") as f_idea:
                idea_text = f_idea.read()
        else:
            idea_md_path = osp.join(base_folder, "idea.md")
            if osp.exists(idea_md_path):
                with open(idea_md_path, "r") as f_idea:
                    idea_text = f_idea.read()
        
        # Load summaries if available
        # Potentially some of the stages might not have been run, so we handle missing files gracefully
        summary_files = [
            ("logs/0-run/baseline_summary.json", "BASELINE_SUMMARY"),
            ("logs/0-run/research_summary.json", "RESEARCH_SUMMARY"),
            ("logs/0-run/ablation_summary.json", "ABLATION_SUMMARY"),
        ]
        loaded_summaries = {}
        for fname, key in summary_files:
            path = osp.join(base_folder, fname)
            if osp.exists(path):
                try:
                    with open(path, "r") as f:
                        loaded_summaries[key] = json.load(f)
                except json.JSONDecodeError:
                    print(
                        f"Warning: {fname} is not valid JSON. Using empty data for {key}."
                    )
                    loaded_summaries[key] = {}
            else:
                loaded_summaries[key] = {}
        
        # Convert them to one big JSON string for context
        combined_summaries_str = json.dumps(loaded_summaries, indent=2)
        
        notebook_file = osp.join(notebook_folder, "lab_notebook.md")
        notebook_pdf = osp.join(notebook_folder, "lab_notebook.pdf")
        if osp.exists(notebook_file):
            with open(notebook_file, "r") as f:
                current_notebook = f.read()
        else:
            current_notebook = ""
        
        # Gather plot filenames from figures/ folder
        figures_dir = osp.join(base_folder, "figures")
        plot_names = []
        if osp.exists(figures_dir):
            for fplot in os.listdir(figures_dir):
                if fplot.lower().endswith(".png"):
                    plot_names.append(fplot)
                    
        # Load aggregator script to include in the prompt
        aggregator_path = osp.join(base_folder, "auto_plot_aggregator.py")
        aggregator_code = ""
        if osp.exists(aggregator_path):
            with open(aggregator_path, "r") as fa:
                aggregator_code = fa.read()
        else:
            aggregator_code = "No aggregator script found."
        
        # Generate VLM-based descriptions but do not overwrite plot_names
        try:
            vlm_client, vlm_model = create_vlm_client("gpt-4o-2024-05-13")
            desc_map = {}
            for pf in plot_names:
                ppath = osp.join(figures_dir, pf)
                if not osp.exists(ppath):
                    continue
                img_dict = {
                    "images": [ppath],
                    "caption": "No direct caption",
                }
                review_data = generate_vlm_img_review(img_dict, vlm_model, vlm_client)
                if review_data:
                    desc_map[pf] = review_data.get(
                        "Img_description", "No description found"
                    )
                else:
                    desc_map[pf] = "No description found"

            # Prepare a string listing all figure descriptions in order
            plot_descriptions_list = []
            for fname in plot_names:
                desc_text = desc_map.get(fname, "No description found")
                plot_descriptions_list.append(f"{fname}: {desc_text}")
            plot_descriptions_str = "\n".join(plot_descriptions_list)
        except Exception:
            print("EXCEPTION in VLM figure description generation:")
            print(traceback.format_exc())
            plot_descriptions_str = "No descriptions available."
        
        # Construct final prompt for model, placing the figure descriptions alongside the plot list
        client, client_model = create_client(model)

        combined_prompt = notebook_prompt.format(
            idea_text=idea_text,
            summaries=combined_summaries_str,
            aggregator_code=aggregator_code,
            plot_list=", ".join(plot_names),
            current_notebook=current_notebook,
            plot_descriptions=plot_descriptions_str,
        )

        response, msg_history = get_response_from_llm(
            prompt=combined_prompt,
            client=client,
            model=client_model,
            system_message=notebook_system_message_template,
            print_debug=False,
        )
        
        notebook_code_match = re.search(r"```markdown(.*?)```", response, re.DOTALL)
        if not notebook_code_match:
            return False
        
        updated_notebook = notebook_code_match.group(1).strip()
        
        def save_notebook(notebook_content):
            with open(notebook_file, "w") as f:
                f.write(notebook_content)
        
            print(f"Lab notebook successfully saved")
        
        save_notebook(updated_notebook)
        
        # Multiple reflection loops 
        for i in range(n_notebook_reflections):
            with open(notebook_file, "r") as f:
                current_notebook = f.read()
                
            reflection_prompt = f"""
Now let's reflect and identify any issues (including but not limited to):
1) Is the writing clear?
2) Have we included all relevant details from the summaries without hallucinating?

Please provide a revised complete notebook or repeat the same if no changes are needed.
Return the entire file in full, with no unfilled placeholders!
Do not hallucinate any details!

If you believe you are done, simply say: "I am done".
"""

            reflection_response, msg_history = get_response_from_llm(
                prompt=reflection_prompt,
                client=client,
                model=client_model,
                system_message=notebook_system_message_template,
                msg_history=msg_history,
                print_debug=False,
            )
        
            if "I am done" in reflection_response:
                print(
                    "LLM indicated it is done with reflections. Exiting reflection loop."
                )
                break
        
            reflection_code_match = re.search(r"```markdown(.*?)```", response, re.DOTALL)
            if not reflection_code_match:
                return False
        
            if reflection_code_match:
                reflected_notebook = reflection_code_match.group(1).strip()
                
                if reflected_notebook != current_notebook:
                    save_notebook(reflected_notebook)
                else:
                    print(f"No changes in reflection step {i+1}.")
                    break
            else:
                print(f"No valid notebook found in reflection step {i+1}.")
                break
        
        return osp.exists(notebook_file)
        
    except Exception:
        print("EXCEPTION in create_lab_notebook:")
        print(traceback.format_exc())
        return False
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create lab notebook for the summary of AI Scientist experiments.")
    parser.add_argument("--folder", type=str, help="Project folder", required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for citation collection (small model).",
    )
    parser.add_argument(
        "--notebook-reflections",
        type=int,
        default=3,
        help="Number of reflection steps for the final notebook writeup.",
    )    
    args = parser.parse_args()

    try:
        success = create_lab_notebook(
            base_folder=args.folder,
            model=args.model,
            n_notebook_reflections=args.notebook_reflections,
        )
        if not success:
            print("Writeup process did not complete successfully.")
    except Exception:
        print("EXCEPTION in main:")
        print(traceback.format_exc())
