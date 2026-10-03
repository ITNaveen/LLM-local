#!/bin/bash
OUTPUT="llm_app_context.txt"
echo "=== LLM APP CODEBASE STRUCT ==" > $OUTPUT
find . -maxdepth 3 -not -path '*/.*' -not -name "*.tar.gz" >> $OUTPUT
echo -e "\n=== SOURCE CODE ===\n" >> $OUTPUT

# Grabs common backend/frontend file types
find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.json" -o -name "*.go" \) \
    -not -path '*/.*' -not -name "*.tar.gz" | while read -r file; do
    echo "--- START_FILE: $file ---" >> $OUTPUT
    cat "$file" >> $OUTPUT
    echo -e "\n--- END_FILE: $file ---\n" >> $OUTPUT
done
echo "Pack complete: $OUTPUT"
