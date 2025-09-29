#!/bin/bash

# Sync Wiki Content - Use Lecture Files Directly
# This script creates wiki pages that directly use lecture content

set -e

echo "🚀 Syncing wiki with direct lecture file references..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first:"
    echo "   brew install gh"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "🔐 Please authenticate with GitHub first:"
    echo "   gh auth login"
    exit 1
fi

# Get repository info
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "📚 Syncing wiki for repository: $REPO"

# Enable wiki (this requires API access)
echo "🔧 Ensuring wiki is enabled..."
gh api repos/$REPO --method PATCH --field has_wiki=true

# Clone the wiki repository
WIKI_DIR="wiki-temp"
echo "📥 Cloning wiki repository..."

# Remove existing temp directory if it exists
rm -rf $WIKI_DIR

git clone "https://github.com/$REPO.wiki.git" $WIKI_DIR

cd $WIKI_DIR

# Function to create wiki page that directly uses lecture file
create_wiki_from_lecture() {
    local lecture_num=$1
    local wiki_name=$2
    local lecture_file="../lectures/lecture-${lecture_num}.md"
    local wiki_file="${wiki_name}.md"
    
    if [ -f "$lecture_file" ]; then
        echo "📝 Creating ${wiki_name} from lecture-${lecture_num}..."
        
        # Just remove the Jekyll front matter, keep everything else
        sed '1,/^---$/d; /^---$/d' "$lecture_file" > "$wiki_file"
        
    else
        echo "⚠️  Lecture file not found: $lecture_file"
    fi
}

# Create Home page
echo "📝 Creating Home page..."
cat > Home.md << 'EOF'
# Marvelous MLOps Course Wiki

Welcome to the course wiki! This contains the complete course content and supplementary information.

## 📚 Course Lectures

- [Week 1: Introduction to MLOps & Development Environment](Week-1-Introduction)
- [Week 2: MLflow Experiment Tracking & Model Registry](Week-2-Experiment-Tracking)
- [Week 3: Feature Engineering & DynamoDB Integration](Week-3-Feature-Engineering)
- [Week 4: Model Serving Endpoints](Week-4-Model-Serving)
- [Week 5: Databricks Workflows & MLOps Pipeline](Week-5-Workflows)
- [Week 6: Model Monitoring & Drift Detection](Week-6-Monitoring)

## 🛠️ Quick Links

- [Course Overview](Course-Overview)
- [Setup Guide](Setup-Guide)
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)

---
*This wiki contains the complete course content. For the formatted course website, visit: [Course Documentation](https://end-to-end-mlops-databricks-3.github.io/course-code-hub3)*
EOF

# Create simple overview page
echo "📝 Creating Course Overview..."
cat > Course-Overview.md << 'EOF'
# Course Overview

## Marvelous MLOps: End-to-End MLOps with Databricks

### Course Information
- **Schedule**: Weekly lectures on Wednesdays 16:00-18:00 CET
- **Runtime**: Databricks 15.4 LTS (Python 3.11)
- **Dataset**: House Price Dataset from Kaggle
- **Delivery**: Weekly deliverables with PR-based submissions

### Learning Path
1. **Week 1**: Introduction to MLOps & Development Environment
2. **Week 2**: MLflow Experiment Tracking & Model Registry
3. **Week 3**: Feature Engineering & DynamoDB Integration
4. **Week 4**: Model Serving Endpoints
5. **Week 5**: Databricks Workflows & MLOps Pipeline
6. **Week 6**: Model Monitoring & Drift Detection

### Submission Process
1. Create feature branch for deliverable
2. Implement solution with your dataset
3. Create PR to main branch
4. Code review and CI pipeline approval
5. Final submissions by **June 18th** (Demo Day)
EOF

# Use lecture files directly for weekly content
create_wiki_from_lecture 1 "Week-1-Introduction"
create_wiki_from_lecture 2 "Week-2-Experiment-Tracking"
create_wiki_from_lecture 3 "Week-3-Feature-Engineering"
create_wiki_from_lecture 4 "Week-4-Model-Serving"
create_wiki_from_lecture 5 "Week-5-Workflows"
create_wiki_from_lecture 6 "Week-6-Monitoring"

# Copy static content if it exists
if [ -f "../wiki-content/Setup-Guide.md" ]; then
    cp "../wiki-content/Setup-Guide.md" "Setup-Guide.md"
    echo "📝 Copied Setup Guide"
fi

if [ -f "../wiki-content/Troubleshooting.md" ]; then
    cp "../wiki-content/Troubleshooting.md" "Troubleshooting.md"
    echo "📝 Copied Troubleshooting"
fi

if [ -f "../wiki-content/FAQ.md" ]; then
    cp "../wiki-content/FAQ.md" "FAQ.md"
    echo "📝 Copied FAQ"
fi

# Check if there are any changes
if git diff --quiet; then
    echo "✅ No changes detected. Wiki is up to date."
else
    echo "💾 Committing wiki updates..."
    git add .
    git commit -m "Sync wiki with lecture content - $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin master
    echo "✅ Wiki updated successfully!"
fi

cd ..
rm -rf $WIKI_DIR

echo "🌐 Visit your wiki at: https://github.com/$REPO/wiki"
