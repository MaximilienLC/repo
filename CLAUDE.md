Do not add a signature to commit messages.

Always think.

# Execution

## Podman

Run python scripts using podman with image `maximilienlc/repo:rocm` and the following flags:

```
--rm
--cap-add=SYS_PTRACE
--security-opt seccomp=unconfined
--ipc=host
--device=/dev/dxg
-v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so -v /opt/rocm/lib/libhsa-runtime64.so.1:/opt/rocm/lib/libhsa-runtime64.so.1
```

## Logging

Always write out the logs in data/ (at the root of the repo)

Use this template command:

```
bash -c 'TIMESTAMP=$(date +%Y%m%d_%H%M%S) && mkdir -p data/$TIMESTAMP && podman run ... -e OUTPUT_DIR=/workspace/data/$TIMESTAMP ... python /workspace/SCRIPT_PATH 2>&1 | tee data/$TIMESTAMP/SCRIPT_NAME.log'
```

# Code Generation

## Instructions

When outputting code, always output the smallest self-contained bit of code at a time. Try your best to never have it exceed 10 lines.

You will at times propose a serie of code writes/changes.
In such instances, follow the following template:

- Before any such serie: write out `CODE BLOCK SERIE:` followed by an explanation of what you're about to write, why and how. 

- Before any code generation, write out `CODE BLOCK:` followed by a similar explanation. Make sure to only detail what will be written in the single following code block. If you need multiple code blocks to write out some somewhat self-contained logic, you can add a `CODE BLOCK SUBSERIE:` and detail the more high-level description there.

Do not include any instructions given to you in these explanations.

## Miscellaneous

Always use context7 when writing code

Don't add this `#!/usr/bin/env python3` type of header on python files

Always add a new line at the end of files.