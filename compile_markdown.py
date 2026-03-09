import datetime as dt
import markdown as md
import os
import re
import shutil as sh
import sys

def compile(markdown_filepath: str, publish_date: str | None = None) -> str:

    article_md = load_markdown(markdown_filepath)
    title, article_md = split_title_from_article(article_md)
    article_html = compile_markdown_to_html(article_md)

    template_head, template_tail = get_webpage_template()
    read_time = estimate_reading_time_minutes(article_md)
    article_header = create_article_header(title, read_time, publish_date)
    article_body = create_article_body(article_html)

    html_complete = '\n'.join([template_head, article_header, article_body, template_tail])
    directory = get_directory(markdown_filepath)
    write_html(directory, html_complete)
    return directory

def load_markdown(markdown_filepath: str) -> str:
    with open(markdown_filepath, 'rt') as file:
        markdown_text = file.read()
    return markdown_text

def split_title_from_article(text: str) -> tuple[str, str]:
    split = text.split('\n\n')
    title = split[0]
    article = '\n\n'.join(split[1:])
    match = re.match(r'# (.*)', title)
    if match is None:
        raise Exception(f'Could not parse title from "{title}"')
    title = match.group(1)
    return title, article

def compile_markdown_to_html(markdown_text: str) -> str:
    return md.markdown(markdown_text, extensions=['fenced_code'])
    
def tabs(num_tabs: int, *, tab_size: int = 4) -> str:
    return ' ' * (num_tabs * tab_size)

def get_webpage_template(filepath: str = 'pages/blogs/blog_template/index.html',
                            article_div_header: str = tabs(3) + '<div class="article">\n') -> tuple[str, str]:
    with open(filepath, 'rt') as file:
        template = file.read()
    split = template.split(article_div_header)
    head = split[0]
    tail = split[1]
    head = head + article_div_header
    return head, tail

def estimate_reading_time_minutes(text: str, words_per_minute: int = 250) -> int:
    word_count = len(re.split(r"\s+", text.strip()))
    est_time_to_read = round(word_count / words_per_minute)
    est_time_to_read = max(1, est_time_to_read)
    return est_time_to_read

def get_date(date_str: str | None) -> dt.date:
    if date_str is None:
        return dt.date.today()
    else:
        return dt.date.strptime(date_str, '%Y-%m-%d')
    
def create_article_header(title: str, read_time_minutes: int, pub_date: str | None) -> str:
    date_str = get_date(pub_date).strftime('%B %d, %Y')
    article_header = []
    article_header.append(tabs(4) + '<div class="article-header">\n')
    article_header.append(tabs(5) + f'<h1>{title}</h1>\n')
    article_header.append(tabs(5) + f'<h2>by Sean Bush | {date_str} | {read_time_minutes} min read</h2>\n')
    article_header.append(tabs(4) + '</div>\n')
    article_header = ''.join(article_header)
    return article_header

def create_article_body(article: str) -> str:
    article_body = []
    for line in article.split('\n'):
        article_body.append(tabs(4) + line + '\n')
    article_body = ''.join(article_body)
    return article_body

def get_directory(markdown_filepath: str) -> str:
    directory = re.match(r'(.*\/)[a-zA-Z0-9\-\_\(\)]*.md', markdown_filepath)
    if directory is None:
        raise Exception(f'Could not create HTML filename from MD filename: {markdown_filepath}')
    directory = directory.group(1)
    return directory

def write_html(directory: str, text: str) -> None:
    filepath = directory + 'index.html'
    with open(filepath, 'wt') as file:
        file.write(text)

def create_favicon(target_dir: str, source_path: str = 'pages/blogs/blog_template/favicon.ico') -> None:
    target_path = os.path.join(target_dir, 'favicon.ico')
    sh.copy(source_path, target_path)

if __name__ == '__main__':
    md_filepath = sys.argv[1]
    pub_date = sys.argv[2] if len(sys.argv) >= 3 else None
    directory = compile(md_filepath, pub_date)
    create_favicon(directory)
    print(f'Page compiled.')