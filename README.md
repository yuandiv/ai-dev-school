# AI大模型应用开发知识库

## 项目简介

这是一个专注于 AI 大模型应用开发的知识库，旨在帮助开发者构建 AI 大模型应用的核心能力体系。内容涵盖开发框架、模型训练、Agent开发、RAG技术、模型部署等多个技术领域。

**在线访问地址**：<https://devdiv.github.io/school>

## 核心特色

- **基础知识** - 大模型原理、提示工程、API使用、AI编程工具，构建AI开发基础
- **开发框架** - LangChain、LlamaIndex、AutoGen、HuggingFace，掌握主流开发框架
- **模型训练** - LoRA微调、数据工程、模型蒸馏，深入模型训练与微调
- **Agent开发** - Function Calling、MCP协议、自主规划、记忆能力，构建智能体系统
- **RAG技术** - Embeddings、向量数据库、混合检索、RAG调优，掌握检索增强生成
- **工程实践** - 模型部署、低代码平台、项目实战，实现全栈工程能力

## 内容模块

### AI大模型开发

- **基础知识** - 大模型原理、提示工程、API使用
- **开发框架** - LangChain、LlamaIndex、AutoGen、HuggingFace
- **模型训练** - 微调原理、数据工程、模型蒸馏
- **Agent开发** - Function Calling、MCP协议、自主规划、记忆能力
- **RAG技术** - Embeddings、向量数据库、多模态处理、RAG调优
- **模型部署** - 企业级部署、vLLM、SGLang、性能优化
- **低代码平台** - Coze、Dify、系统集成
- **项目实战** - 企业知识库、OpenManus、AI质检、ChatBI

## 快速开始

### 环境要求

- Node.js >= 18.0.0
- npm >= 9.0.0

### 本地运行

```bash
# 克隆项目
git clone https://github.com/devdiv/school.git
cd school

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问 `http://localhost:8080/school/` 查看本地文档。

### 构建部署

```bash
# 构建静态文件
npm run build

# 部署到 GitHub Pages
npm run deploy
```

## 技术栈

- **框架**: [VuePress 2.0](https://vuepress.vuejs.org/)
- **构建工具**: [Vite](https://vitejs.dev/)
- **主题**: VuePress Default Theme
- **插件**:
  - PWA 支持
  - Google Analytics
  - 全文搜索
  - 代码高亮 (PrismJS)
  - Markdown图表 (Mermaid/Markmap)

## 项目结构

```
school/
├── docs/                    # 文档源文件
│   ├── .vuepress/          # VuePress 配置
│   │   ├── components/     # 自定义组件
│   │   ├── public/         # 静态资源
│   │   ├── styles/         # 样式文件
│   │   └── config.ts       # 配置文件
│   └── ai-llm-dev/         # AI大模型开发
│       ├── basics/         # 基础知识
│       ├── frameworks/     # 开发框架
│       ├── training/       # 模型训练
│       ├── agent/          # Agent开发
│       ├── rag/            # RAG技术
│       ├── deployment/     # 模型部署
│       ├── low-code/       # 低代码平台
│       └── projects/       # 项目实战
├── package.json            # 项目配置
├── deploy.sh               # 部署脚本
└── README.md               # 项目说明
```

## 贡献指南

欢迎贡献内容、提出问题或建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 文档编写规范

- 使用 Markdown 格式编写文档
- 文件名使用小写字母和连字符
- 每个目录应包含 `README.md` 作为索引页
- 图片等资源文件放在对应目录的 `images` 文件夹中

## 许可证

本项目采用 [MIT](LICENSE) 许可证。

## 相关链接

- **在线文档**: <https://devdiv.github.io/school>
- **GitHub 仓库**: <https://github.com/devdiv/school>
- **问题反馈**: <https://github.com/devdiv/school/issues>
