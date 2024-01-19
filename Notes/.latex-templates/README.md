# Example of main file:

```latex
\documentclass[11pt]{report}

% some definitions for the title page
\newcommand{\reporttitle}{example}
\newcommand{\reportdescription}{example description}

% load some definitions and default packages
\input{../.latex-templates/includes}

\begin{document}

% Include the title page
\input{../.latex-templates/titlepage}

% Start of new text
Hello World

\end{document}
```