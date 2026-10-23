# Getting started

## Goals

This repository will be used to run the Scattering Transform on sea surface
temperature (SST) data, characterizing the multi-scale, non-Gaussian structure of
the SST field.  We will generate the analyses, metrics, and diagnostics to share
with the community.  We will also compare ocean model output (LLC4320_v2) to 
satellite data.

## Prompts

1. Read this file.  Execute the 1st task under "Claude/CLAUDE.md file"
2. Read this file.  Execute the 1st task under "Claude/Skills"
3. Read this file.  Execute the 1st task under "Claude/Settings"
4. Read this file.  Execute the 1st task under "Basic start up"


## Claude

### CLAUDE.md file

1. Please generate a basic CLAUDE.md file for this project.  Have it indicate:

- I will perform git commands
- Add to the CLAUDE.md file:  If you do any calculation, generate it as a python script and write it to disk so that I can add it to the Repository.
- Add to the CLAUDE.md file:  If you need to run Python, use the "ocean14" conda environment.
- Add to the CLAUDE.md file:  Prompt docs for this repository live in prompts/ (not claude_prompts/ as in the sibling repositories).

### Skills

1. Copy over the skills/ files from the `IOPtics` repository.

### Settings

1. Copy over the settings.json file from the `IOPtics` repository.  Copy the policy
   (the allow/deny/ask structure), not the accumulated path-specific entries that
   reference IOPtics' own scratchpad and test files.

## Basic start up

1. Generate the basic files that one needs for a Python GitHub repository, e.g. a file for dependencies.  Examine the other Repositories in Oceanography/python to see how I tend to organize things.  Also, make a suggestion of which of their settings.json files to copy into this one.


## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs
