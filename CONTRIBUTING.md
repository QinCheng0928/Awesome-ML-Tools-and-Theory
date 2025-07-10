# 📝 Git Commit Message Convention

To maintain a clean and understandable commit history, it's recommended to follow a standardized commit message format. This helps with version control, code reviews, and automation.

## 🔖 Commit Message Format

```
<type>: <short description>
```

- `type`: The category of the change (see the table below).
- `short description`: A brief summary of the change written in the imperative mood.

## 📚 Type Definitions

| Type       | Description                                                  |
| ---------- | ------------------------------------------------------------ |
| `feat`     | **Feature**: Introduces a new feature or enhancement, such as a new endpoint or module. |
| `fix`      | **Bug Fix**: Fixes a bug or defect in the code to restore correct behavior. |
| `docs`     | **Documentation**: Changes related to documentation only, such as `README` updates, inline comments, or API docs. |
| `style`    | **Style**: Changes in code formatting, indentation, spaces, semicolons, etc. that do **not** affect the logic or behavior of the code. |
| `refactor` | **Refactor**: Code restructuring that improves readability or structure **without** changing external behavior or functionality. |
| `perf`     | **Performance**: Improvements related to performance, such as reducing memory usage or improving response time. |
| `test`     | **Test**: Adding or modifying tests such as unit, integration, or end-to-end tests, without affecting production code. |
| `revert`   | **Revert**: Reverts a previous commit, typically used to undo unintended changes. |

## ✅ Examples

```
git commit -m "feat: add user registration API"
git commit -m "fix: fix token validation error during login"
git commit -m "refactor: improve logic for querying user list"
git commit -m "docs: update API documentation"
git commit -m "style: format code and adjust indentation"
git commit -m "perf: improve performance of user list query"
git commit -m "test: add unit test for user registration API"
git commit -m "revert: revert user list caching logic"
```

