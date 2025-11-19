# GitHub 上传指南

## 📋 上传前检查清单

### ✅ 已完成

- [x] `.gitignore` 已配置（排除敏感文件）
- [x] `README.md` 完整文档已创建
- [x] `requirements.txt` 依赖列表已创建
- [x] `env.example` 配置模板已创建
- [x] 测试脚本和过期代码已清理

### ⚠️ 请确认

- [ ] **`.env` 文件不会被上传**（已在 .gitignore 中）
- [ ] **API Key 不在代码中**（都在 .env 文件中）
- [ ] **数据库文件不会被上传**（已在 .gitignore 中）
- [ ] **敏感信息已移除**（检查代码中是否有硬编码密钥）

---

## 🚀 上传步骤

### 方法 1: 使用命令行（推荐）

#### 步骤 1: 检查状态

```bash
git status
```

#### 步骤 2: 添加所有更改

```bash
# 添加所有新文件和修改
git add .

# 或选择性添加
git add README.md
git add QUICKSTART.md
git add requirements.txt
git add env.example
git add .gitignore
git add agno_agent.py
git add rag_tools.py
git add crag_core.py
git add crag_layer.py
git add document_processor.py
git add persistent_vector_store.py
git add upload_documents.py
```

#### 步骤 3: 提交更改

```bash
git commit -m "docs: 添加完整文档和清理项目

- 添加初学者友好的 README.md
- 添加快速开始指南 QUICKSTART.md
- 添加 requirements.txt 和 .gitignore
- 清理测试脚本和过期文档
- 修复 CRAG 实现（Action 判定、search_knowledge）
- 添加 DISABLE_FAST_PATH 配置选项"
```

#### 步骤 4: 推送到 GitHub

```bash
# 推送到 main 分支
git push origin main

# 如果远程分支不同，使用：
git push origin main:main
```

---

### 方法 2: 使用 GitHub Desktop（图形界面）

1. **打开 GitHub Desktop**
2. **选择仓库**: Agno-RAG
3. **查看更改**: 左侧会显示所有修改
4. **填写提交信息**:
   ```
   docs: 添加完整文档和清理项目

   - 添加初学者友好的 README.md
   - 添加快速开始指南 QUICKSTART.md
   - 添加 requirements.txt 和 .gitignore
   - 清理测试脚本和过期文档
   - 修复 CRAG 实现（Action 判定、search_knowledge）
   - 添加 DISABLE_FAST_PATH 配置选项
   ```
5. **点击 "Commit to main"**
6. **点击 "Push origin"** 推送到 GitHub

---

## 📝 提交信息格式建议

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>: <description>

[optional body]

[optional footer]
```

**类型 (type)**:
- `feat`: 新功能
- `fix`: 修复 bug
- `docs`: 文档更新
- `refactor`: 代码重构
- `clean`: 清理代码
- `chore`: 其他更改

**示例**:

```bash
# 文档更新
git commit -m "docs: 添加完整的 README 和快速开始指南"

# 功能修复
git commit -m "fix: 修复 CRAG Action 判定逻辑（使用最高分而非平均分）"

# 代码清理
git commit -m "clean: 删除测试脚本和过期文档"
```

---

## 🔒 安全检查

### 检查敏感信息

在上传前，检查以下内容：

```bash
# 检查是否有 API Key 泄露（替换为您的 API Key 前缀）
grep -r "sk-" . --exclude-dir=.git --exclude-dir=venv

# 检查是否有硬编码密钥
grep -ri "api_key.*=" . --exclude-dir=.git --exclude-dir=venv | grep -v ".env"
grep -ri "password.*=" . --exclude-dir=.git --exclude-dir=venv | grep -v ".env"

# 检查 .env 文件是否被跟踪
git ls-files | grep -E "\.env$|\.env\."
```

**如果发现敏感信息**：
1. 立即从代码中删除
2. 如果已经提交过，使用 `git filter-branch` 或 BFG Repo-Cleaner 清理历史
3. 如果已推送到 GitHub，立即更换密钥

---

## 📦 首次上传到新仓库

如果您要在 GitHub 上创建**新仓库**：

### 步骤 1: 在 GitHub 创建仓库

1. 登录 GitHub
2. 点击右上角 "+" → "New repository"
3. 填写信息：
   - **Repository name**: `Agno-RAG`
   - **Description**: `基于 CRAG 的智能文档问答系统 | CRAG-based Intelligent Document Q&A System`
   - **Visibility**: Public / Private
   - **不要**勾选 "Initialize with README"（因为您已经有了）

### 步骤 2: 连接本地仓库

如果仓库已存在，检查远程地址：

```bash
git remote -v
```

如果没有远程仓库，添加：

```bash
# 替换 <your-username> 为您的 GitHub 用户名
git remote add origin https://github.com/<your-username>/Agno-RAG.git

# 或使用 SSH
git remote add origin git@github.com:<your-username>/Agno-RAG.git
```

### 步骤 3: 推送代码

```bash
# 推送并设置上游分支
git push -u origin main
```

---

## 🌟 GitHub 仓库设置建议

### 添加仓库描述和主题

在 GitHub 仓库页面：
1. 点击 "⚙️ Settings"
2. 在 "Repository details" 中：
   - **Description**: `基于 CRAG 的智能文档问答系统 | CRAG-based Intelligent Document Q&A System`
   - **Topics**: 
     ```
     rag
     crag
     retrieval-augmented-generation
     llm
     document-qa
     agno
     lancedb
     semantic-search
     python
     ```

### 添加 README 徽章（可选）

在 README.md 顶部添加（已在 README.md 中）：

```markdown
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Agno](https://img.shields.io/badge/Agno-Framework-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
```

### 添加 GitHub Actions（可选）

创建 `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python -m pytest  # 如果有测试
```

---

## ✅ 上传后检查

### 检查上传的文件

访问您的 GitHub 仓库，确认：

- [ ] `README.md` 正确显示
- [ ] `.gitignore` 已生效（敏感文件未上传）
- [ ] `requirements.txt` 存在
- [ ] `env.example` 存在
- [ ] 代码文件完整
- [ ] PDF 文件（如果包含）已上传

### 检查敏感文件

确认以下文件**未**出现在 GitHub：

- [ ] `.env` - ✅ 已在 .gitignore
- [ ] `*.db` - ✅ 已在 .gitignore
- [ ] `agno_knowledge.db` - ✅ 已在 .gitignore
- [ ] `venv/` - ✅ 已在 .gitignore
- [ ] `__pycache__/` - ✅ 已在 .gitignore

---

## 🎉 完成！

上传成功后：

1. **分享链接**: `https://github.com/<your-username>/Agno-RAG`
2. **添加 Stars**: 给自己的项目点个 ⭐
3. **添加 Releases**: 创建第一个 Release 标签
4. **添加 License**: 在根目录添加 LICENSE 文件（可选）

---

## 📚 后续操作

### 创建 Release

```bash
# 创建标签
git tag -a v1.0.0 -m "初始版本：完整 CRAG 实现"

# 推送标签
git push origin v1.0.0
```

然后在 GitHub 仓库页面 → Releases → "Create a new release"。

### 添加 LICENSE 文件（可选）

```bash
# 创建 MIT License
touch LICENSE
```

编辑 LICENSE 文件，添加 MIT License 模板。

---

## 🆘 遇到问题？

### 问题 1: 推送被拒绝

```bash
# 错误: failed to push some refs
# 解决: 先拉取远程更改
git pull origin main --rebase
git push origin main
```

### 问题 2: 大文件上传失败

```bash
# 错误: remote: error: File is too large
# 解决: 检查大文件，添加到 .gitignore 或使用 Git LFS
git ls-files -z | xargs -0 du -sh | sort -rh | head -20
```

### 问题 3: 提交历史混乱

```bash
# 清理提交历史（谨慎使用）
git reset --soft HEAD~N  # N 是要撤销的提交数
git commit --amend -m "新的提交信息"
git push origin main --force  # 危险操作！
```

---

**准备好了吗？开始上传吧！** 🚀

