#!/bin/bash
set -e

# Release preparation script for train-station
# Usage: ./scripts/prepare-release.sh <version>
# Example: ./scripts/prepare-release.sh 0.1.4

if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.4"
    echo ""
    echo "This script will:"
    echo "1. Validate environment and version format"
    echo "2. Update version in Cargo.toml"
    echo "3. Generate changelog entries from git log"
    echo "4. Run tests to ensure everything works"
    echo "5. Prepare commit and tag commands"
    exit 1
fi

VERSION=$1

echo "=== Train Station Release Preparation ==="
echo "Target version: $VERSION"
echo ""

# Validate version format (semantic versioning)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid version format '$VERSION'"
    echo "Expected format: MAJOR.MINOR.PATCH (e.g., 0.1.4)"
    exit 1
fi

# Verify we're on master branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "master" ]; then
    echo "Error: Must be on master branch to create release"
    echo "Current branch: $BRANCH"
    echo ""
    echo "Switch to master first:"
    echo "  git checkout master"
    echo "  git pull origin master"
    exit 1
fi

# Verify working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean"
    echo "Please commit or stash your changes first:"
    git status --short
    exit 1
fi

# Verify we're up to date with remote
echo "Checking if master is up to date..."
git fetch origin master
LOCAL=$(git rev-parse master)
REMOTE=$(git rev-parse origin/master)
if [ "$LOCAL" != "$REMOTE" ]; then
    echo "Error: Local master is not up to date with origin/master"
    echo "Please pull the latest changes:"
    echo "  git pull origin master"
    exit 1
fi

# Check for unmerged feature branches
echo "Checking for unmerged feature branches..."
unmerged_branches=$(git branch -r --no-merged master | grep -E 'origin/(feat|fix|perf)/' | head -5 || true)
if [ -n "$unmerged_branches" ]; then
    echo "Warning: Found unmerged feature branches:"
    echo "$unmerged_branches"
    echo ""
    read -p "Continue with release? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Release cancelled"
        exit 1
    fi
fi

# Get current version from Cargo.toml
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT_VERSION"
echo "Target version: $VERSION"

# Check if version is newer
if [ "$CURRENT_VERSION" = "$VERSION" ]; then
    echo "Error: Version $VERSION is the same as current version"
    echo "Please specify a newer version"
    exit 1
fi

# Update version in Cargo.toml
echo ""
echo "=== Updating Cargo.toml ==="
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
echo "Updated version in Cargo.toml to $VERSION"

# Generate changelog entries from git log
echo ""
echo "=== Generating Changelog Entries ==="

# Get the last tag to determine commit range
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [ -n "$LAST_TAG" ]; then
    COMMIT_RANGE="$LAST_TAG..HEAD"
    echo "Generating changelog from $LAST_TAG to HEAD"
else
    # If no tags exist, use all commits
    FIRST_COMMIT=$(git rev-list --max-parents=0 HEAD)
    COMMIT_RANGE="$FIRST_COMMIT..HEAD"
    echo "Generating changelog from first commit to HEAD (no previous tags)"
fi

# Create temporary changelog content
TEMP_CHANGELOG=$(mktemp)
echo "## [$VERSION] - $(date +%Y-%m-%d)" > "$TEMP_CHANGELOG"
echo "" >> "$TEMP_CHANGELOG"

# Parse conventional commits and categorize
echo "### Added" >> "$TEMP_CHANGELOG"
FEAT_COMMITS=$(git log --pretty=format:"%s" $COMMIT_RANGE 2>/dev/null | grep "^feat" | sed 's/^feat[^:]*: /- /' || true)
if [ -n "$FEAT_COMMITS" ]; then
    echo "$FEAT_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No new features" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Fixed" >> "$TEMP_CHANGELOG"
FIX_COMMITS=$(git log --pretty=format:"%s" $COMMIT_RANGE 2>/dev/null | grep "^fix" | sed 's/^fix[^:]*: /- /' || true)
if [ -n "$FIX_COMMITS" ]; then
    echo "$FIX_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No bug fixes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Performance" >> "$TEMP_CHANGELOG"
PERF_COMMITS=$(git log --pretty=format:"%s" $COMMIT_RANGE 2>/dev/null | grep "^perf" | sed 's/^perf[^:]*: /- /' || true)
if [ -n "$PERF_COMMITS" ]; then
    echo "$PERF_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No performance improvements" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Documentation" >> "$TEMP_CHANGELOG"
DOCS_COMMITS=$(git log --pretty=format:"%s" $COMMIT_RANGE 2>/dev/null | grep "^docs" | sed 's/^docs[^:]*: /- /' || true)
if [ -n "$DOCS_COMMITS" ]; then
    echo "$DOCS_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No documentation changes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

echo "### Other Changes" >> "$TEMP_CHANGELOG"
OTHER_COMMITS=$(git log --pretty=format:"%s" $COMMIT_RANGE 2>/dev/null | grep -v "^feat\|^fix\|^perf\|^docs\|^chore.*release" | sed 's/^/- /' || true)
if [ -n "$OTHER_COMMITS" ]; then
    echo "$OTHER_COMMITS" >> "$TEMP_CHANGELOG"
else
    echo "- No other changes" >> "$TEMP_CHANGELOG"
fi
echo "" >> "$TEMP_CHANGELOG"

# Update CHANGELOG.md by inserting new version after [Unreleased]
if [ -f "CHANGELOG.md" ]; then
    echo "Updating CHANGELOG.md..."
    
    # Create temporary file for updated changelog
    UPDATED_CHANGELOG=$(mktemp)
    
    # Copy everything up to and including the [Unreleased] section
    sed -n '1,/^## \[Unreleased\]/p' CHANGELOG.md > "$UPDATED_CHANGELOG"
    
    # Reset the Unreleased section
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
    echo "### Security" >> "$UPDATED_CHANGELOG"
    echo "- " >> "$UPDATED_CHANGELOG"
    echo "" >> "$UPDATED_CHANGELOG"
    
    # Add the new version section
    cat "$TEMP_CHANGELOG" >> "$UPDATED_CHANGELOG"
    
    # Add all existing version sections (skip the Unreleased section)
    sed -n '/^## \[Unreleased\]/,/^## \[.*\] - /{ /^## \[Unreleased\]/d; /^## \[.*\] - /,$p }' CHANGELOG.md >> "$UPDATED_CHANGELOG"
    
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

### Security
- 

EOF
    cat "$TEMP_CHANGELOG" >> "CHANGELOG.md"
fi

# Clean up temporary file
rm "$TEMP_CHANGELOG"

# Show the generated changelog section
echo ""
echo "=== Generated Changelog Section ==="
sed -n "/^## \[$VERSION\]/,/^## \[/{ /^## \[/{ /$VERSION/p; /$VERSION/!q }; /^## \[/!p }" CHANGELOG.md

# Run tests to make sure everything works
echo ""
echo "=== Running Tests ==="
echo "Running core library tests..."
if cargo test -p train-station --lib; then
    echo "✓ Core tests passed"
else
    echo "✗ Core tests failed"
    echo "Please fix test failures before releasing"
    exit 1
fi

# Check if code compiles in release mode
echo "Checking release build..."
if cargo build --release; then
    echo "✓ Release build successful"
else
    echo "✗ Release build failed"
    echo "Please fix build errors before releasing"
    exit 1
fi

# Prepare final instructions
echo ""
echo "=== Release Preparation Complete! ==="
echo ""
echo "Changes made:"
echo "- Updated Cargo.toml version to $VERSION"
echo "- Updated CHANGELOG.md with new version section"
echo "- All tests passed"
echo ""
echo "Next steps:"
echo "1. Review the changes:"
echo "   git diff"
echo ""
echo "2. Commit the release:"
echo "   git add Cargo.toml CHANGELOG.md"
echo "   git commit -m \"chore: release version $VERSION\""
echo ""
echo "3. Create and push the tag:"
echo "   git tag v$VERSION"
echo "   git push origin master --tags"
echo ""
echo "4. The GitHub Actions will automatically:"
echo "   - Run full test suite"
echo "   - Create GitHub release with auto-generated notes"
echo "   - Handle any missing releases for existing tags"
echo ""
echo "Release preparation completed successfully! 🚀"
