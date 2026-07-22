# Research Standards
We want to run experiments with high velocity but we also need those experiments to have lasting value, which means the results need to be communicated with enough context that we can understand them months or years in the future. Here we outline our standards for how to structure your project and share research results so they are maximally useful, both now and in the future.
Getting Started

## Github
create a repo using the AGI-Template repository (if it doesn’t exist).
add the ais team as an admin
add the ReviewNB app (if you can, if not ask @Mike Vaiana or @Diogo de Lucena )
Share your experiment results via PR (use draft mode if you want early feedback).
ask your team who you should add as a reviewer, or just tag everyone and we’ll sort it out.
we use ReviewNB to allow reviews of rendered notebooks 🔥
## Compute
If you need access to EC2, runpod, API keys, or any other type of compute dependencies then reach out to Mike Vaiana or Stijn Servaes right away.
## Drive
Each project has a dedicated folder in Google Drive to keep information organized and easy to navigate. The structure is consistent across projects so team members can quickly find what they need.
# Project Folder

Each project has a dedicated folder in Google Drive to keep information organized and easy to navigate. The structure is consistent across projects so team members can quickly find what they need.
Info – Project information, kickoff materials, background notes. Acts as a “dump” for initial context.
Reports – Contains monthly reports (1–2 pages each), stored with clear dating (e.g., ProjectName August 2025).
Progress Updates – Weekly or stakeholder updates, stored chronologically (e.g., ProjectName 2025-07-10).
Rolling internal deck containing the experimental updates that can be used in stakeholder slide decks or the internal team sync deck.
Non-Technical Summary – A single doc that’s updated monthly with the high-level state of the project for non-technical collaborators, grant writing, and onboarding. Versioning is used within this document.
Submissions – Folder that contains specific documentation and final versions of attempted/failed submissions. This includes rebuttals, papers, comments, checklists, and any additional information.
```
ProjectName/
├── Info/
├── Reports/
│   ├── ProjectName July 2025
│   └── ProjectName August 2025
├── Progress Updates/
│   ├── ProjectName 2025-07-10
│   └── ProjectName 2025-07-17
├── ProjectName (rolling internal slidedeck)
└── Non-Technical Summary
├── Submissions/
│   ├── ICLR 2024
│   └── NeurIPS 2025
```
This structure ensures continuity, makes onboarding easier, and keeps both technical and non-technical outputs visible and accessible.
# Github Structure

We have found through trial and error that the project structure that tends to offer the right trade-off of flexibility and isolation is the one below. For a research project, create a repo with an experiments folder.  Each Linear issue (i.e. research question) should have its own folder which contains all the code to run the associated experiments and analyze the results.   


These “research” folders should have at least one Jupyter notebook that reports the results.  It's fine to do all of your work directly inside of a Jupyter notebook.  If you do your work in experiment scripts, then use a notebook to aggregate and report results.  Name the folder with the Linear issue ID and a brief description as shown below
```
root
|- README.md
|- experiments
	|- AGI-244-some-experiment
		|- results.ipynb
		|- run_experiment_1.py
		|- run_experiment_2.py
		|- logs/
	|- AGI-321-other-experiement
		|- experiment.ipynb
|- spikes
|- src
	|- package_name
	|- tests
```
Sometimes a project benefits from shared or repeated code.  One option is to copy paste code from one experiment to another.  This is actually fine - it's much faster and easier to make changes to the local copy.  However, if you find yourself needing some utilities that pretty much never change, then you might want to factor that out into a package that you can import.  This would be an engineering story - make sure that if you are doing this it's because you have a strong need and it will greatly speed up or improve the quality of your work. 
# Reports

We expect all experimental results to be reported through a Jupyter notebook. We expect a certain level of quality of the notebook that helps with the following
The PR reviewer can better understand the context of the experiment and better interpret the results
In the future (months or years) we may want to revisit old results and a clean notebook with proper context facilitates this
The notebook should follow this template
```
<Report Name>
<story id (if applicable)>
<date XXXX-XX-XX >
Introduction
Some context on what our hypothesis is, why we are asking it, any historical context that is relevant, and how we plan to test the hypothesis
…<all code and results>…
Conclusion
```
A conclusion that summarizes the results and ties them back to the original research question. When applicable include possible new questions that have come up in the conclusion.
Additionally, all figures should have a “caption” of text that explains the figures so they can be interpreted without reading the code. If a figure has no value, remove it from the notebook, if it does have value, then it should have a caption.
❗Important
Each report should correspond to exactly one linear issue.  This means that each linear issue should have a dedicated folder.

# Reviews
When you’ve completed a report, open a PR on Github and tag the appropriate people for review (likely your TPM or other researchers on the team).  

Reviewers.  It is expected that you will review the results of the PR, in other words, you will mostly review the report.  Some things to ask yourself when reviewing
Is the report high quality?  Is it easy to read and does it follow the guidelines above?
Are the results surprising – is it possible there is a bug or gotcha happening?
Are the experiments well designed?  Do they actually help answer the question?
Are there any obvious extensions (follow up experiments) that would be very helpful to answer the question?
Do the experiments faithfully follow the core/periphery?  Did we achieve high velocity (tried lots of things)?
Is the main question asked by the report answered? 
 If not, do we need a totally different approach? 
 Are there only null results?  If yes, do we need to explore more or do we need to rethink our approach conceptually?
What new research questions or goals do we have based on the results?


Also make sure to get a gut check on the code itself.  Is it well organized and free from any red flags?  Was there any concern if bugs from the report - if yes can you clearly identify them or identify and sketchy code you have questions about?
# Spikes

Sometimes we run spikes to quickly test an idea. In this case all the rules are out the window. You should sprint towards a result and not worry about any formatting or formalities above. However, the spike still provides us information, so try to communicate clearly the main point of the spike and the outcome.
Tips for doing research

Read these first
Michael Bernstein’s Velocity in R&D
Ethan Perez’s Tips
Jacob Steinhardt's Post on Reducing Uncertainty
Always
What is the question we are asking?
Why are we asking it? Is this the right question?
What type of experiment would reduce uncertainty fastest?
Try to show your idea is wrong as fast as possible!
Be collaborative, share your ideas, challenge each other
Try lots of things quickly - it’s fine if they all fail
When presenting
What is the minimal context to understand the question and motivation?
What is your experiment/approach to answering this question?
What is the result and how does this answer the original question?
It’s fine if the answer is: we need to investigate more
When reviewing
What is the question we were trying to answer?
Does the actual implementation and metrics answer this question? What would a reviewer from a journal or conference say?
Are there any valuable next steps? If yes, make sure you are asking the right questions
Other
Try to embrace YAGNI especially if you tend to over-engineer. This can greatly accelerate the pace at which you can execute experiments.
Use AI tools as much as possible, they can help accelerate brain storming or writing code scaffolding for experiments and more. If you aren’t comfortable with this, now is a great time to practice
