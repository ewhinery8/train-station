#!/bin/bash
set -e

# Simple release preparation script for train-station
# Usage: ./scripts/prepare-release.sh <version>

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
error() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}Info: $1${NC}"
}

success() {
    echo -e "${GREEN}Success: $1${NC}"
}

warning() {
    echo -e "${YELLOW}Warning: $1${NC}"
}

# Check if version argument provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.1.4"
    echo ""
    echo "This script will:"
    echo "  1. Validate the environment and version format"
    echo "  2. Update version in Cargo.toml"
    echo "  3. Generate release notes from git log"
    echo "  4. Update CHANGELOG.md"
    echo "  5. Run tests to ensure everything works"
    echo "  6. Provide instructions for completing the release"
    exit 1
fi

VERSION=$1

info "Preparing release $VERSION..."

# Validate version format (basic semver check)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    error "Version must be in semver format (e.g., 0.1.4)"
fi

# Verify we're on main branch
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    error "Must be on main branch to create release (currently on: $BRANCH)"
fi

# Verify working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    error "Working directory is not clean. Please commit or stash changes first."
fi

# Check if we're up to date with remote
git fetch origin main
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" != "$REMOTE" ]; then
    error "Local main branch is not up to date with origin/main. Please pull latest changes."
fi

# Check if tag already exists
if git tag -l | grep -q "^v$VERSION$"; then
    error "Tag v$VERSION already exists"
fi

# Get current version from Cargo.toml
CURRENT_VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
info "Current version: $CURRENT_VERSION"
info "New version: $VERSION"

# Verify this is a version bump (basic check)
if [ "$CURRENT_VERSION" = "$VERSION" ]; then
    error "New version ($VERSION) is the same as current version ($CURRENT_VERSION)"
fi

# Check for unmerged feature branches (warning only)
info "Checking for unmerged feature branches..."
unmerged_branches=$(git branch -r --no-merged main | grep -E 'origin/(feat|fix|perf)/' | head -5 || true)
if [ -n "$unmerged_branches" ]; then
    warning "Found unmerged feature branches:"
    echo "$unmerged_branches"
    echo ""
    read -p "Continue with release? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        info "Release cancelled"
        exit 0
    fi
fi

# Update version in Cargo.toml
info "Updating version in Cargo.toml..."
sed -i "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
success "Updated Cargo.toml version to $VERSION"

# Generate release notes from git log
info "Generating release notes from git log..."
LAST_TAG=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

if [ -n "$LAST_TAG" ]; then
    info "Generating changes since $LAST_TAG"
    COMMIT_RANGE="$LAST_TAG..HEAD"
else
    info "No previous tags found, generating all commits"
    COMMIT_RANGE="HEAD"
fi

# Create temporary file for release notes
TEMP_NOTES=$(mktemp)

# Generate release notes by parsing conventional commits
echo "# Release Notes for v$VERSION" > "$TEMP_NOTES"
echo "" >> "$TEMP_NOTES"
echo "## Changes" >> "$TEMP_NOTES"
echo "" >> "$TEMP_NOTES"

# Parse commits and categorize them
git log --pretty=format:"%s" $COMMIT_RANGE | while IFS= read -r commit; do
    case "$commit" in
        feat:*|feat\(*\):*)
            echo "### Added" >> "$TEMP_NOTES.added" 2>/dev/null || echo "### Added" > "$TEMP_NOTES.added"
            echo "- ${commit#feat*: }" >> "$TEMP_NOTES.added"
            ;;
        fix:*|fix\(*\):*)
            echo "### Fixed" >> "$TEMP_NOTES.fixed" 2>/dev/null || echo "### Fixed" > "$TEMP_NOTES.fixed"
            echo "- ${commit#fix*: }" >> "$TEMP_NOTES.fixed"
            ;;
        perf:*|perf\(*\):*)
            echo "### Performance" >> "$TEMP_NOTES.perf" 2>/dev/null || echo "### Performance" > "$TEMP_NOTES.perf"
            echo "- ${commit#perf*: }" >> "$TEMP_NOTES.perf"
            ;;
        docs:*|docs\(*\):*)
            echo "### Documentation" >> "$TEMP_NOTES.docs" 2>/dev/null || echo "### Documentation" > "$TEMP_NOTES.docs"
            echo "- ${commit#docs*: }" >> "$TEMP_NOTES.docs"
            ;;
        refactor:*|refactor\(*\):*)
            echo "### Changed" >> "$TEMP_NOTES.changed" 2>/dev/null || echo "### Changed" > "$TEMP_NOTES.changed"
            echo "- ${commit#refactor*: }" >> "$TEMP_NOTES.changed"
            ;;
        test:*|test\(*\):*)
            echo "### Testing" >> "$TEMP_NOTES.test" 2>/dev/null || echo "### Testing" > "$TEMP_NOTES.test"
            echo "- ${commit#test*: }" >> "$TEMP_NOTES.test"
            ;;
        chore:*|chore\(*\):*)
            # Skip chore commits in release notes unless they're important
            if [[ "$commit" == *"release"* ]] || [[ "$commit" == *"version"* ]]; then
                continue
            fi
            echo "### Maintenance" >> "$TEMP_NOTES.chore" 2>/dev/null || echo "### Maintenance" > "$TEMP_NOTES.chore"
            echo "- ${commit#chore*: }" >> "$TEMP_NOTES.chore"
            ;;
        *)
            echo "### Other" >> "$TEMP_NOTES.other" 2>/dev/null || echo "### Other" > "$TEMP_NOTES.other"
            echo "- $commit" >> "$TEMP_NOTES.other"
            ;;
    esac
done

# Combine all sections in order
for section in added fixed perf changed docs test chore other; do
    if [ -f "$TEMP_NOTES.$section" ]; then
        echo "" >> "$TEMP_NOTES"
        cat "$TEMP_NOTES.$section" >> "$TEMP_NOTES"
        rm -f "$TEMP_NOTES.$section"
    fi
done

# Update CHANGELOG.md
info "Updating CHANGELOG.md..."
DATE=$(date +%Y-%m-%d)

# Create backup of CHANGELOG.md
cp CHANGELOG.md CHANGELOG.md.backup

# Create new CHANGELOG.md with the new version
{
    # Keep header
    sed -n '1,/^## \[Unreleased\]/p' CHANGELOG.md
    
    # Add empty unreleased section
    echo ""
    echo "### Added"
    echo "- "
    echo ""
    echo "### Changed"
    echo "- "
    echo ""
    echo "### Fixed"
    echo "- "
    echo ""
    echo "### Performance"
    echo "- "
    echo ""
    echo "### Security"
    echo "- "
    echo ""
    
    # Add new version section
    echo "## [$VERSION] - $DATE"
    echo ""
    
    # Add generated release notes (skip the header)
    tail -n +4 "$TEMP_NOTES"
    echo ""
    
    # Add rest of changelog (skip unreleased section)
    sed -n '/^## \[Unreleased\]/,$p' CHANGELOG.md | tail -n +2 | sed '/^### Added/,/^### Security/{/^### Security/!d;}' | tail -n +2
    
} > CHANGELOG.md.new

mv CHANGELOG.md.new CHANGELOG.md
rm -f CHANGELOG.md.backup
success "Updated CHANGELOG.md"

# Clean up temp files
rm -f "$TEMP_NOTES"

# Run tests to make sure everything works
info "Running tests..."
if cargo test -p train-station --lib --quiet; then
    success "All tests passed"
else
    error "Tests failed. Please fix issues before releasing."
fi

# Show preview of changes
info "Preview of changes:"
echo ""
echo "=== CHANGELOG.md (new section) ==="
sed -n "/^## \[$VERSION\]/,/^## \[/p" CHANGELOG.md | head -n -1
echo ""

echo "=== Cargo.toml version ==="
grep "^version = " Cargo.toml
echo ""

# Final instructions
success "Release preparation complete!"
echo ""
info "Next steps:"
echo "1. Review the changes above"
echo "2. If everything looks good, run:"
echo "   git add Cargo.toml CHANGELOG.md"
echo "   git commit -m \"chore: release version $VERSION\""
echo "   git tag v$VERSION"
echo "   git push origin main --tags"
echo ""
echo "3. GitHub Actions will handle:"
echo "   - Running comprehensive tests"
echo "   - Creating GitHub release"
echo "   - Publishing to crates.io"
echo ""
warning "If you need to make changes, the release can be cancelled at any time before pushing the tag."
