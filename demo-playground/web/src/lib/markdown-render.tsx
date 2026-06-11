import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

export const MARKDOWN_PLUGINS = {
  remark: [remarkGfm, remarkMath] as const,
  rehype: [rehypeKatex] as const,
};

interface MarkdownRenderProps {
  source: string;
  className?: string;
}

/** Shared markdown + LaTeX renderer (notebook cells and cell outputs). */
export default function MarkdownRender({ source, className = 'prose prose-sm max-w-none' }: MarkdownRenderProps) {
  return (
    <div className={className}>
      <ReactMarkdown remarkPlugins={[...MARKDOWN_PLUGINS.remark]} rehypePlugins={[...MARKDOWN_PLUGINS.rehype]}>
        {source}
      </ReactMarkdown>
    </div>
  );
}
