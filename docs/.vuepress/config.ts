import { viteBundler } from '@vuepress/bundler-vite';
import { defaultTheme } from '@vuepress/theme-default';
import { searchPlugin } from '@vuepress/plugin-search';
import { markdownChartPlugin } from '@vuepress/plugin-markdown-chart';

export default {
  bundler: viteBundler(),
  theme: defaultTheme({
    logo: '/images/hero.png',
    themePlugins: {
      prismjs: { lineNumbers: true },
    },
    lastUpdatedText: '上次更新',
    contributorsText: '贡献者',
    tip: '提示',
    warning: '注意',
    danger: '警告',
    notFound: ['这里什么都没有', '我们怎么到这来了？', '这是一个 404 页面', '看起来我们进入了错误的链接'],
    backToHome: '返回首页',
    openInNewWindow: '在新窗口打开',
    toggleDarkMode: '切换夜间模式',
    toggleSidebar: '切换侧边栏',
    docsRepo: 'devdiv/school',
    editLinks: true,
    editLinkText: '在线编辑文档',
    docsDir: 'docs',
    docsBranch: 'master',
    navbar: [
      { text: '首页', link: '/' },
      { text: 'AI大模型开发', children: [
        { text: '基础知识', link: '/ai-llm-dev/basics/' },
        { text: '开发框架', link: '/ai-llm-dev/frameworks/' },
        { text: '模型训练', link: '/ai-llm-dev/training/' },
        { text: 'Agent开发', link: '/ai-llm-dev/agent/' },
        { text: 'RAG技术', link: '/ai-llm-dev/rag/' },
        { text: '模型部署', link: '/ai-llm-dev/deployment/' },
        { text: '评估与测试', link: '/ai-llm-dev/evaluation/' },
        { text: '低代码平台', link: '/ai-llm-dev/low-code/' },
        { text: '项目实战', link: '/ai-llm-dev/projects/' }
      ]}
    ],
    layouts: {
      Layout: '@/layouts/Layout.vue'
    },
    sidebar: {
      '/ai-llm-dev/': [{ text: 'AI大模型开发', collapsable: true, children: ['', 'basics/', 'basics/llm-principles', 'basics/prompt-engineering', 'basics/api-usage', 'basics/transformer', 'frameworks/', 'frameworks/langchain', 'frameworks/llamaindex', 'frameworks/autogen', 'frameworks/huggingface', 'training/', 'training/lora-finetune', 'training/data-engineering', 'agent/', 'agent/mcp-protocol', 'agent/multi-agent', 'rag/', 'rag/advanced-rag', 'rag/vector-database', 'deployment/', 'deployment/model-quantization', 'evaluation/', 'low-code/', 'projects/'] }],
      '/ai-llm-dev/basics/': [{ text: '基础知识', collapsable: true, children: ['', 'llm-principles', 'prompt-engineering', 'api-usage', 'transformer'] }],
      '/ai-llm-dev/frameworks/': [{ text: '开发框架', collapsable: true, children: ['', 'langchain', 'llamaindex', 'autogen', 'huggingface'] }],
      '/ai-llm-dev/agent/': [{ text: 'Agent开发', collapsable: true, children: ['', 'mcp-protocol', 'multi-agent'] }],
      '/ai-llm-dev/rag/': [{ text: 'RAG技术', collapsable: true, children: ['', 'advanced-rag', 'vector-database'] }],
      '/ai-llm-dev/training/': [{ text: '模型训练', collapsable: true, children: ['', 'lora-finetune', 'data-engineering'] }],
      '/ai-llm-dev/deployment/': [{ text: '模型部署', collapsable: true, children: ['', 'model-quantization'] }],
      '/ai-llm-dev/evaluation/': [{ text: '评估与测试', collapsable: true, children: [''] }],
    }
  }),
  lang: 'zh-CN',
  title: 'AI大模型应用开发知识库',
  description: '构建AI大模型应用开发核心能力体系',
  base: '/school/',
  head: [
    ['link', { rel: 'apple-touch-icon', href: '/icons/apple-touch-icon.png' }],
    ['link', { rel: 'icon', href: 'images/favicon.ico' }],
    ['link', { rel: 'manifest', href: '/manifest.json' }],
    ['meta', { name: 'theme-color', content: '#ffffff' }],
  ],
  plugins: [
    ['@vuepress/plugin-pwa'],
    ['@vuepress/plugin-google-analytics', { id: 'UA-109340118-1' }],
    searchPlugin({
      locales: { '/': { placeholder: '搜索' } },
      maxSuggestions: 10,
      isSearchable: (page) => page.path !== '/',
      getExtraFields: (page) => page.frontmatter.tags ?? [],
    }),
    markdownChartPlugin({
      mermaid: true,
      markmap: true,
    }),
  ],
};
