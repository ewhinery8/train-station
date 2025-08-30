#!/bin/bash
set -e

# Branch creation helper script for train-station
# Usage: ./scripts/create-branch.sh <type> <description>
# Example: ./scripts/create-branch.sh feat matrix-multiplication

if [ $# -ne 2 ]; then
    echo "Usage: $0 <type> <description>"
    echo ""
    echo "Branch Types:"
    echo "  feat      - New features or enhancements"
    echo "  fix       - Bug fixes"
    echo "  perf      - Performance improvements"
    echo "  docs      - Documentation changes"
    echo "  refactor  - Code refactoring"
    echo "  test      - Adding or updating tests"
    echo "  chore     - Maintenance tasks"
    echo ""
    echo "Examples:"
    echo "  $0 feat tensor-broadcasting"
    echo "  $0 fix memory-leak-in-autograd"
    echo "  $0 perf simd-optimization"
    echo "  $0 docs api-documentation-update"
    exit 1
fi

TYPE=$1
DESCRIPTION=$2

# Validate branch type
case $TYPE in
    feat|fix|perf|docs|refactor|test|chore)
        ;;
    *)
        echo "Error: Invalid branch type '$TYPE'"
        echo "Valid types: feat, fix, perf, docs, refactor, test, chore"
        exit 1
        ;;
esac

# Validate description format
if [[ ! $DESCRIPTION =~ ^[a-z0-9-]+$ ]]; then
    echo "Error: Description must be lowercase with hyphens only"
    echo "Invalid: '$DESCRIPTION'"
    echo "Valid examples: 'tensor-broadcasting', 'memory-leak-fix', 'simd-optimization'"
    exit 1
fi

BRANCH_NAME="$TYPE/$DESCRIPTION"

echo "Creating branch: $BRANCH_NAME"

# Verify we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Verify working directory is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: Working directory is not clean"
    echo "Please commit or stash your changes first:"
    git status --short
    exit 1
fi

# Verify we're on master branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "master" ]; then
    echo "Error: Must be on master branch to create new branch"
    echo "Current branch: $CURRENT_BRANCH"
    echo ""
    echo "Switch to master first:"
    echo "  git checkout master"
    echo "  git pull origin master"
    exit 1
fi

# Check if branch already exists locally
if git show-ref --verify --quiet refs/heads/$BRANCH_NAME; then
    echo "Error: Branch '$BRANCH_NAME' already exists locally"
    echo "Use a different description or delete the existing branch:"
    echo "  git branch -D $BRANCH_NAME"
    exit 1
fi

# Check if branch exists on remote
if git ls-remote --exit-code --heads origin $BRANCH_NAME >/dev/null 2>&1; then
    echo "Error: Branch '$BRANCH_NAME' already exists on remote"
    echo "Use a different description or fetch and checkout the existing branch:"
    echo "  git fetch origin $BRANCH_NAME:$BRANCH_NAME"
    echo "  git checkout $BRANCH_NAME"
    exit 1
fi

# Ensure master is up to date
echo "Updating master branch..."
git fetch origin master
git merge --ff-only origin/master

# Create and checkout the new branch
echo "Creating and switching to branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

echo ""
echo "Branch '$BRANCH_NAME' created successfully!"
echo ""
echo "Next steps:"
echo "1. Make your changes"
echo "2. Commit with conventional message format:"
echo "   git commit -m \"$TYPE: <description>\""
echo "3. Push the branch:"
echo "   git push -u origin $BRANCH_NAME"
echo "4. Create a pull request on GitHub"
echo ""
echo "Branch naming guidelines:"
echo "- Use lowercase letters only"
echo "- Use hyphens to separate words"
echo "- Be descriptive but concise"
echo "- Match the commit type you'll use"
