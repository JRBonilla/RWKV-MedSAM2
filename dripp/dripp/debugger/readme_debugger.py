"""README viewer tab for the DRIPP debugger app."""

import tkinter.font as tkfont
import webbrowser

from .common import *


class ReadmeDebuggerMixin:
    """
    Provide a rich README viewer tab for the debugger app.

    Args:
        None.

    Returns:
        None.
    """

    def _readme_path(self):
        """
        Return the README path shown by the debugger.

        Args:
            None.

        Returns:
            str: Absolute README path.
        """
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
        )

    def _build_readme_tab(self):
        """
        Build a read-only README viewer tab.

        Args:
            None.

        Returns:
            None.
        """
        self.readme_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.readme_tab, text="README")

        controls = ttk.Frame(self.readme_tab)
        controls.pack(fill="x", padx=5, pady=5)

        ttk.Button(controls, text="Reload", command=self._load_readme).pack(
            side="right",
            padx=2,
        )
        ttk.Button(controls, text="Open File", command=self._open_readme_file).pack(
            side="right",
            padx=2,
        )

        self.readme_text = scrolledtext.ScrolledText(
            self.readme_tab,
            wrap="word",
            font=("Segoe UI", 10),
            padx=14,
            pady=10,
        )
        self.readme_text.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self._configure_readme_tags()
        self._load_readme()

    def _configure_readme_tags(self):
        """
        Configure text styles used by the README renderer.

        Args:
            None.

        Returns:
            None.
        """
        base_font = tkfont.nametofont("TkDefaultFont")
        fixed_font = tkfont.nametofont("TkFixedFont")
        self.readme_fonts = {
            "body": base_font.copy(),
            "h1": base_font.copy(),
            "h2": base_font.copy(),
            "h3": base_font.copy(),
            "bold": base_font.copy(),
            "italic": base_font.copy(),
            "code": fixed_font.copy(),
        }
        self.readme_fonts["h1"].configure(size=18, weight="bold")
        self.readme_fonts["h2"].configure(size=14, weight="bold")
        self.readme_fonts["h3"].configure(size=12, weight="bold")
        self.readme_fonts["bold"].configure(weight="bold")
        self.readme_fonts["italic"].configure(slant="italic")
        self.readme_fonts["code"].configure(size=10)

        self.readme_text.tag_configure(
            "h1",
            font=self.readme_fonts["h1"],
            spacing1=8,
            spacing3=8,
        )
        self.readme_text.tag_configure(
            "h2",
            font=self.readme_fonts["h2"],
            spacing1=8,
            spacing3=6,
        )
        self.readme_text.tag_configure(
            "h3",
            font=self.readme_fonts["h3"],
            spacing1=6,
            spacing3=4,
        )
        self.readme_text.tag_configure("bold", font=self.readme_fonts["bold"])
        self.readme_text.tag_configure("italic", font=self.readme_fonts["italic"])
        self.readme_text.tag_configure(
            "code",
            font=self.readme_fonts["code"],
            background="#f2f4f7",
            lmargin1=18,
            lmargin2=18,
            spacing1=3,
            spacing3=3,
        )
        self.readme_text.tag_configure(
            "inline_code",
            font=self.readme_fonts["code"],
            background="#eef1f5",
        )
        self.readme_text.tag_configure(
            "quote",
            foreground="#555555",
            lmargin1=18,
            lmargin2=18,
        )
        self.readme_text.tag_configure("bullet", lmargin1=18, lmargin2=36)
        self.readme_text.tag_configure("link", foreground="#0563c1", underline=True)

    def _load_readme(self):
        """
        Load README markdown into the viewer.

        Args:
            None.

        Returns:
            None.
        """
        path = self._readme_path()
        self.readme_text.configure(state="normal")
        self.readme_text.delete("1.0", tk.END)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            self._show_readme_not_found()
            self.readme_text.configure(state="disabled")
            return

        self._render_readme_markdown(content)
        self.readme_text.configure(state="disabled")

    def _show_readme_not_found(self):
        """
        Display a friendly message when the README cannot be loaded.

        Args:
            None.

        Returns:
            None.
        """
        self.readme_text.insert(tk.END, "README not found", ("h2",))
        self.readme_text.insert(
            tk.END,
            "\n\nThe DRIPP README could not be found in the expected project location.",
        )

    def _render_readme_markdown(self, content):
        """
        Render a small Markdown subset into a tagged Text widget.

        Args:
            content (str): Markdown content.

        Returns:
            None.
        """
        self._readme_links = []
        in_code_block = False

        for line in content.splitlines():
            stripped = line.strip()

            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue

            if in_code_block:
                self.readme_text.insert(tk.END, line + "\n", ("code",))
                continue

            if not stripped:
                self.readme_text.insert(tk.END, "\n")
                continue

            heading = re.match(r"^(#{1,3})\s+(.*)$", line)
            if heading:
                tag = f"h{len(heading.group(1))}"
                self._insert_readme_inline(heading.group(2), (tag,))
                self.readme_text.insert(tk.END, "\n")
                continue

            if re.match(r"^\s*\|.*\|\s*$", line):
                self.readme_text.insert(tk.END, line + "\n", ("code",))
                continue

            if re.match(r"^\s*[-*_]{3,}\s*$", line):
                self.readme_text.insert(tk.END, "\n" + ("-" * 72) + "\n")
                continue

            quote = re.match(r"^\s*>\s?(.*)$", line)
            if quote:
                self._insert_readme_inline(quote.group(1), ("quote",))
                self.readme_text.insert(tk.END, "\n")
                continue

            bullet = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
            if bullet:
                prefix = "  " * (len(bullet.group(1)) // 2) + bullet.group(2) + " "
                self.readme_text.insert(tk.END, prefix, ("bullet",))
                self._insert_readme_inline(bullet.group(3), ("bullet",))
                self.readme_text.insert(tk.END, "\n")
                continue

            self._insert_readme_inline(line)
            self.readme_text.insert(tk.END, "\n")

    def _insert_readme_inline(self, text, base_tags=()):
        """
        Insert inline Markdown spans into the README viewer.

        Args:
            text (str): Text to insert.
            base_tags (tuple): Tags applied to every inserted span.

        Returns:
            None.
        """
        pattern = re.compile(
            r"(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(\[[^\]]+\]\([^)]+\))"
        )
        pos = 0
        for match in pattern.finditer(text):
            if match.start() > pos:
                self.readme_text.insert(tk.END, text[pos:match.start()], base_tags)

            token = match.group(0)
            if token.startswith("`"):
                self.readme_text.insert(tk.END, token[1:-1], base_tags + ("inline_code",))
            elif token.startswith("**"):
                self.readme_text.insert(tk.END, token[2:-2], base_tags + ("bold",))
            elif token.startswith("*"):
                self.readme_text.insert(tk.END, token[1:-1], base_tags + ("italic",))
            else:
                label, target = re.match(r"\[([^\]]+)\]\(([^)]+)\)", token).groups()
                self._insert_readme_link(label, target, base_tags)

            pos = match.end()

        if pos < len(text):
            self.readme_text.insert(tk.END, text[pos:], base_tags)

    def _insert_readme_link(self, label, target, base_tags=()):
        """
        Insert a clickable README link.

        Args:
            label (str): Link text.
            target (str): Link destination.
            base_tags (tuple): Tags applied to the link.

        Returns:
            None.
        """
        tag = f"readme_link_{len(self._readme_links)}"
        self._readme_links.append(target)
        self.readme_text.insert(tk.END, label, base_tags + ("link", tag))
        self.readme_text.tag_bind(
            tag,
            "<Button-1>",
            lambda _event, url=target: self._open_readme_link(url),
        )
        self.readme_text.tag_bind(
            tag,
            "<Enter>",
            lambda _event: self.readme_text.configure(cursor="hand2"),
        )
        self.readme_text.tag_bind(
            tag,
            "<Leave>",
            lambda _event: self.readme_text.configure(cursor=""),
        )

    def _open_readme_link(self, target):
        """
        Open a link from the README viewer.

        Args:
            target (str): URL or README-relative path.

        Returns:
            None.
        """
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            webbrowser.open(target)
            return

        base_dir = os.path.dirname(self._readme_path())
        path = os.path.abspath(os.path.join(base_dir, target))
        if os.path.exists(path):
            webbrowser.open(path)
            return

        messagebox.showerror("Link Not Found", f"The path does not exist:\n{path}")

    def _open_readme_file(self):
        """
        Open the README file in the system-associated application.

        Args:
            None.

        Returns:
            None.
        """
        path = self._readme_path()
        if not os.path.exists(path):
            messagebox.showerror("README Not Found", "README not found.")
            return

        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Unable to Open", f"Could not open README:\n{e}")
