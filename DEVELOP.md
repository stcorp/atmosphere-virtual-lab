# update online documentation

- pydoc-markdown -m avl -m avl.vis > index.md
- copy index.md to gh-pages branch.
- check that avl/__init__.py docstring has made it in or copy it yourself (not sure why it's left out)
- commit and push
- github actions tab should show that documentation is updated
