#!/bin/bash
set -e

# Release preparation script for train-station
# Usage: ./scripts/prepare-release.sh
# 
# This script should be run from a feature branch before merging to master.
# It will prepare the changelog and provide commit/tag templates for release.

if [ $# -ne 0 ]; then
    echo "Usage: $0"
    echo ""
    echo "This script should be run from a feature branch before merging to master."
    echo "It will:"
    echo "1. Check that version has been updated compared to master"
    echo "2. Generate changelog entries from branch changes"
    echo "3. Update CHANGELOG.md with new version section"
    echo "4. Provide commit and tag templates for release"
    echo ""
    echo "Note: This script only modifies CHANGELOG.md and provides templates."
    echo "No git commands are executed automatically."
    exit 1
fi

echo "=== Train Station Release Preparation ==="
echo ""

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" = "master" ]; then
    echo "Error: This script should be run from a feature branch, not master"
    echo "Current branch: $CURRENT_BRANCH"
    echo ""
    echo "Please run this from your feature branch before merging to master."
    exit 1
fi

echo "Current branch: $CURRENT_BRANCH"

# Fetch latest master to ensure we have up-to-date comparison
echo "Fetching latest master for comparison..."
git fetch origin master

# Get version from current branch and master
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
MASTER_VERSION=$(git show origin/master:Cargo.toml | grep '^version = ' | head -1 | sed 's/version = "\(.*\)"/\1/')

echo "Master version: $MASTER_VERSION"
echo "Current version: $CURRENT_VERSION"

# Check if version has been updated
if [ "$CURRENT_VERSION" = "$MASTER_VERSION" ]; then
    echo ""
    echo "Error: Version has not been updated compared to master"
    echo "Please update the version in Cargo.toml before running this script."
    echo ""
    echo "Current version in both branches: $CURRENT_VERSION"
    echo "Expected: A newer version than $MASTER_VERSION"
    exit 1
fi

echo ""
echo "Version has been updated: $MASTER_VERSION → $CURRENT_VERSION"

# Generate changelog entries from branch changes
echo ""
echo "=== Generating Changelog Entries ==="

# Get all commits from master to current branch (including staged changes)
COMMIT_RANGE="origin/master..HEAD"
echo "Analyzing changes from master to current branch..."

# Check if there are any commits to analyze
COMMIT_COUNT=$(git rev-list --count $COMMIT_RANGE 2>/dev/null || echo "0")
if [ "$COMMIT_COUNT" -eq 0 ]; then
    echo "Warning: No commits found between master and current branch"
    echo "This might indicate the branch is not ahead of master"
fi

# Create temporary changelog content
TEMP_CHANGELOG=$(mktemp)
echo "## [$CURRENT_VERSION] - $(date +%Y-%m-%d)" > "$TEMP_CHANGELOG"
echo "" >> "$TEMP_CHANGELOG"

# Parse conventional commits and categorize by full commit messages
# Get all commit hashes in reverse order (oldest first for proper changelog order)
COMMIT_HASHES=$(git rev-list --reverse $COMMIT_RANGE 2>/dev/null || true)

# Initialize sections
FEAT_COMMITS=""
FIX_COMMITS=""
PERF_COMMITS=""
DOCS_COMMITS=""
CHORE_COMMITS=""
BREAKING_COMMITS=""
OTHER_COMMITS=""

# Process each commit individually to get full messages
if [ -n "$COMMIT_HASHES" ]; then
    for commit in $COMMIT_HASHES; do
        # Get the commit subject and body separately
        SUBJECT=$(git log --format="%s" -n 1 "$commit")
        RAW_BODY=$(git log --format="%b" -n 1 "$commit")

        # Strip conventional prefix from subject since section headers indicate the type
        STRIPPED_SUBJECT=$(echo "$SUBJECT" | sed -E 's/^[a-z]+(!)?(\([^)]*\))?:[[:space:]]*//')
        COMMIT_ENTRY="- $STRIPPED_SUBJECT"
        
        # Format body as indented bullet list (    - ...), stripping any existing list markers
        if [ -n "$RAW_BODY" ]; then
            BODY_BULLETS=$(echo "$RAW_BODY" \
              | sed 's/\r$//' \
              | awk 'NF' \
              | sed -E 's/^[[:space:]]*[-*][[:space:]]+//' \
              | sed -E 's/^[[:space:]]+//' \
              | sed 's/^/    - /')
            if [ -n "$BODY_BULLETS" ]; then
                COMMIT_ENTRY="$COMMIT_ENTRY"$'\n'"$BODY_BULLETS"
            fi
        fi
        
        # Check for breaking changes first (any commit with !)
        if [[ "$SUBJECT" =~ ! ]]; then
            if [ -n "$BREAKING_COMMITS" ]; then
                BREAKING_COMMITS="$BREAKING_COMMITS"$'\n\n'
            fi
            
            # For breaking changes, extract BREAKING CHANGE: content or use already-stripped subject
            BREAKING_ENTRY=""
            if [[ "$RAW_BODY" =~ BREAKING\ CHANGE:[[:space:]]*(.*) ]]; then
                # Extract text after "BREAKING CHANGE:"
                BREAKING_TEXT="${BASH_REMATCH[1]}"
                BREAKING_ENTRY="- $BREAKING_TEXT"
            else
                # Fallback to stripped subject (prefix already removed above)
                BREAKING_ENTRY="- $STRIPPED_SUBJECT"
            fi
            
            BREAKING_COMMITS="$BREAKING_COMMITS$BREAKING_ENTRY"
        fi
        
        # Categorize based on conventional commit type
        if [[ "$SUBJECT" =~ ^feat[^:]*: ]]; then
            if [ -n "$FEAT_COMMITS" ]; then
                FEAT_COMMITS="$FEAT_COMMITS"$'\n\n'
            fi
            FEAT_COMMITS="$FEAT_COMMITS$COMMIT_ENTRY"
        elif [[ "$SUBJECT" =~ ^fix[^:]*: ]]; then
            if [ -n "$FIX_COMMITS" ]; then
                FIX_COMMITS="$FIX_COMMITS"$'\n\n'
            fi
            FIX_COMMITS="$FIX_COMMITS$COMMIT_ENTRY"
        elif [[ "$SUBJECT" =~ ^perf[^:]*: ]]; then
            if [ -n "$PERF_COMMITS" ]; then
                PERF_COMMITS="$PERF_COMMITS"$'\n\n'
            fi
            PERF_COMMITS="$PERF_COMMITS$COMMIT_ENTRY"
        elif [[ "$SUBJECT" =~ ^docs[^:]*: ]]; then
            if [ -n "$DOCS_COMMITS" ]; then
                DOCS_COMMITS="$DOCS_COMMITS"$'\n\n'
            fi
            DOCS_COMMITS="$DOCS_COMMITS$COMMIT_ENTRY"
        elif [[ "$SUBJECT" =~ ^chore[^:]*: ]] && [[ ! "$SUBJECT" =~ release ]]; then
            if [ -n "$CHORE_COMMITS" ]; then
                CHORE_COMMITS="$CHORE_COMMITS"$'\n\n'
            fi
            CHORE_COMMITS="$CHORE_COMMITS$COMMIT_ENTRY"
        else
            # Handle other commit types (refactor, test, style, etc.)
            if [ -n "$OTHER_COMMITS" ]; then
                OTHER_COMMITS="$OTHER_COMMITS"$'\n\n'
            fi
            OTHER_COMMITS="$OTHER_COMMITS$COMMIT_ENTRY"
        fi
    done
fi

# Add sections to changelog
echo "### Added" >> "$TEMP_CHANGELOG"
if [ -n "$FEAT_COMMITS" ]; then
    echo "$FEAT_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No new features" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Fixed" >> "$TEMP_CHANGELOG"
if [ -n "$FIX_COMMITS" ]; then
    echo "$FIX_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No bug fixes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Performance" >> "$TEMP_CHANGELOG"
if [ -n "$PERF_COMMITS" ]; then
    echo "$PERF_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No performance improvements" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Breaking Changes" >> "$TEMP_CHANGELOG"
if [ -n "$BREAKING_COMMITS" ]; then
    echo "$BREAKING_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No breaking changes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Documentation" >> "$TEMP_CHANGELOG"
if [ -n "$DOCS_COMMITS" ]; then
    echo "$DOCS_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No documentation changes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Maintenance" >> "$TEMP_CHANGELOG"
if [ -n "$CHORE_COMMITS" ]; then
    echo "$CHORE_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No maintenance changes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Other Changes" >> "$TEMP_CHANGELOG"
if [ -n "$OTHER_COMMITS" ]; then
    echo "$OTHER_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No other changes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

# Update CHANGELOG.md by inserting new version after [Unreleased]
echo ""
echo "=== Updating CHANGELOG.md ==="

if [ -f "CHANGELOG.md" ]; then
    echo "Updating CHANGELOG.md..."
    
    # Create temporary file for updated changelog
    UPDATED_CHANGELOG=$(mktemp)
    
    # Find where the Unreleased section ends
    UNRELEASED_END=$(grep -n "^## \[Unreleased\]" CHANGELOG.md | cut -d: -f1)
    FIRST_RELEASE=$(grep -n "^## \[[0-9]" CHANGELOG.md | head -1 | cut -d: -f1)
    
    # Copy header and Unreleased section title
    sed -n "1,${UNRELEASED_END}p" CHANGELOG.md > "$UPDATED_CHANGELOG"
    
    # Add reset Unreleased section
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Added" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Changed" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Fixed" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Performance" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Breaking Changes" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Documentation" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Maintenance" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    echo "### Security" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    
    # Add the new version section
    cat "$TEMP_CHANGELOG" >> "$UPDATED_CHANGELOG"
    
    # Add all existing releases (preserve everything from first release onward)
    if [ -n "$FIRST_RELEASE" ]; then
        sed -n "${FIRST_RELEASE},\$p" CHANGELOG.md >> "$UPDATED_CHANGELOG"
    fi
    
    # Replace the original changelog
    mv "$UPDATED_CHANGELOG" "CHANGELOG.md"
    echo "CHANGELOG.md updated successfully"
else
    echo "Warning: CHANGELOG.md not found, creating new one..."
    cat > "CHANGELOG.md" << EOF
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- 

### Changed
- 

### Fixed
- 

### Performance
- 

### Breaking Changes
- 

### Documentation
- 

### Maintenance
- 

### Security
- 

EOF
    cat "$TEMP_CHANGELOG" >> "CHANGELOG.md"
    echo "✓ CHANGELOG.md created with new version section"
fi

# Clean up temporary file
rm "$TEMP_CHANGELOG"

# Normalize CHANGELOG formatting (bullets/blank lines) recursively
echo "Normalizing CHANGELOG formatting..."
TMP_NORM=$(mktemp)
awk 'NF{blank=0} !NF{if(blank) next; blank=1} {print}' CHANGELOG.md > "$TMP_NORM"
mv "$TMP_NORM" CHANGELOG.md
sed -i -E \
  -e 's/^([[:space:]]*-)[[:space:]]+$/\1 /' \
  -e 's/^  - /    - /' \
  -e 's/^[[:space:]]+$/\n/' \
  CHANGELOG.md

# Show the generated changelog section
echo ""
echo "=== Generated Changelog Section ==="
sed -n "/^## \[$CURRENT_VERSION\]/,/^## \[/{ /^## \[/{ /$CURRENT_VERSION/p; /$CURRENT_VERSION/!q }; /^## \[/!p }" CHANGELOG.md

echo ""
echo "=== Release Preparation Complete! ==="
echo ""
echo "Changes made:"
echo "- Updated CHANGELOG.md with version $CURRENT_VERSION section"
echo ""
echo "=== COMMIT TEMPLATE ==="
echo "Use this commit message for the final commit before merging to master:"
echo ""
echo "chore: release version $CURRENT_VERSION"
echo ""
echo "=== TAG TEMPLATE ==="
echo "Use this tag after merging to master:"
echo ""
echo "v$CURRENT_VERSION"
echo ""
echo "=== NEXT STEPS ==="
echo "1. Review the CHANGELOG.md changes:"
echo "   git diff CHANGELOG.md"
echo ""
echo "2. Commit the changelog (copy and paste):"
echo "   git add CHANGELOG.md"
echo "   git commit -m \"chore: release version $CURRENT_VERSION\""
echo ""
echo "3. Merge this branch to master (via PR or direct merge)"
echo ""
echo "4. After merging to master, create and push the tag:"
echo "   git checkout master"
echo "   git pull origin master"
echo "   git tag v$CURRENT_VERSION"
echo "   git push origin master --tags"
echo ""
echo "5. GitHub Actions will automatically create the release"
echo ""
echo "IMPORTANT: This script only modified CHANGELOG.md."
echo "No other files were changed, and no git commands were executed."
echo ""
echo "Release preparation completed successfully!"
