#!/bin/bash
set -e

# Changelog entry helper script for train-station
# Usage: ./scripts/add-changelog-entry.sh <type> <description>
# Example: ./scripts/add-changelog-entry.sh added "matrix multiplication with SIMD optimization"

if [ $# -ne 2 ]; then
    echo "Usage: $0 <type> <description>"
    echo ""
    echo "Entry Types:"
    echo "  added       - New features or enhancements"
    echo "  changed     - Changes in existing functionality"
    echo "  fixed       - Bug fixes"
    echo "  performance - Performance improvements"
    echo "  security    - Security-related changes"
    echo ""
    echo "Examples:"
    echo "  $0 added \"tensor broadcasting support for arbitrary shapes\""
    echo "  $0 fixed \"memory leak in gradient computation\""
    echo "  $0 performance \"SIMD optimization for element-wise operations\""
    echo "  $0 changed \"API signature for matrix multiplication\""
    echo "  $0 security \"input validation for tensor operations\""
    exit 1
fi

TYPE=$1
DESCRIPTION=$2

# Validate entry type
case $TYPE in
    added|changed|fixed|performance|security)
        ;;
    *)
        echo "Error: Invalid entry type '$TYPE'"
        echo "Valid types: added, changed, fixed, performance, security"
        exit 1
        ;;
esac

# Validate description
if [ -z "$DESCRIPTION" ]; then
    echo "Error: Description cannot be empty"
    exit 1
fi

# Check if CHANGELOG.md exists
if [ ! -f "CHANGELOG.md" ]; then
    echo "Error: CHANGELOG.md not found in current directory"
    echo "Make sure you're in the project root directory"
    exit 1
fi

# Capitalize first letter of type for section matching
SECTION_NAME=$(echo "$TYPE" | sed 's/^./\U&/')

echo "Adding entry to CHANGELOG.md..."
echo "Type: $SECTION_NAME"
echo "Description: $DESCRIPTION"

# Create a temporary file for the updated changelog
TEMP_FILE=$(mktemp)

# Process the changelog line by line
IN_UNRELEASED=false
IN_TARGET_SECTION=false
SECTION_FOUND=false
ENTRY_ADDED=false

while IFS= read -r line; do
    # Check if we're entering the Unreleased section
    if [[ "$line" == "## [Unreleased]" ]]; then
        IN_UNRELEASED=true
        echo "$line" >> "$TEMP_FILE"
        continue
    fi
    
    # Check if we're leaving the Unreleased section (next version section)
    if [[ "$line" =~ ^##[[:space:]]\[.*\][[:space:]]-[[:space:]] ]] && [ "$IN_UNRELEASED" = true ]; then
        IN_UNRELEASED=false
        IN_TARGET_SECTION=false
    fi
    
    # If we're in the Unreleased section, look for our target section
    if [ "$IN_UNRELEASED" = true ] && [[ "$line" == "### $SECTION_NAME" ]]; then
        IN_TARGET_SECTION=true
        SECTION_FOUND=true
        echo "$line" >> "$TEMP_FILE"
        continue
    fi
    
    # If we're leaving our target section (next ### section)
    if [ "$IN_TARGET_SECTION" = true ] && [[ "$line" =~ ^###[[:space:]] ]] && [[ "$line" != "### $SECTION_NAME" ]]; then
        IN_TARGET_SECTION=false
    fi
    
    # If we're in our target section and find an empty entry line, replace it
    if [ "$IN_TARGET_SECTION" = true ] && [[ "$line" == "- " ]] && [ "$ENTRY_ADDED" = false ]; then
        echo "- $DESCRIPTION" >> "$TEMP_FILE"
        ENTRY_ADDED=true
        continue
    fi
    
    # If we're in our target section and this is a non-empty entry, add our entry before it
    if [ "$IN_TARGET_SECTION" = true ] && [[ "$line" =~ ^-[[:space:]].+ ]] && [ "$ENTRY_ADDED" = false ]; then
        echo "- $DESCRIPTION" >> "$TEMP_FILE"
        ENTRY_ADDED=true
    fi
    
    # Copy the current line
    echo "$line" >> "$TEMP_FILE"
    
done < "CHANGELOG.md"

# Check if we found the section and added the entry
if [ "$SECTION_FOUND" = false ]; then
    echo "Error: Could not find '### $SECTION_NAME' section in Unreleased"
    echo "Make sure CHANGELOG.md follows the expected format"
    rm "$TEMP_FILE"
    exit 1
fi

if [ "$ENTRY_ADDED" = false ]; then
    echo "Error: Could not add entry to the $SECTION_NAME section"
    echo "The section might not have the expected format"
    rm "$TEMP_FILE"
    exit 1
fi

# Replace the original file with the updated version
mv "$TEMP_FILE" "CHANGELOG.md"

echo ""
echo "Successfully added changelog entry!"
echo ""
echo "Entry added:"
echo "### $SECTION_NAME"
echo "- $DESCRIPTION"
echo ""
echo "You can now commit this change:"
echo "  git add CHANGELOG.md"
echo "  git commit -m \"docs: update changelog with $TYPE entry\""
echo ""
echo "Or continue adding more entries before committing."
