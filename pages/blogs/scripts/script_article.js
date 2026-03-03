function format_code_blocks() {
    
    code_blocks = document.querySelectorAll('.article pre code');

    for(const block of code_blocks) {

        const inner = block.innerHTML;

        const lines = inner.split('\n');
        const processed_lines = [];

        processed_lines.push(lines[0]);
        for(let i = 1; i < lines.length - 1; i++) {
            processed_lines.push(lines[i].slice(16));
        }

        const processed = processed_lines.join('\n');
        block.innerHTML = processed;

    }

}

format_code_blocks();