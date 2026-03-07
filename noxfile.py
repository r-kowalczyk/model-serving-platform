import nox

# Global options
nox.options.sessions = ("ruff", "mypy", "bandit")
nox.options.reuse_existing_virtualenvs = True
nox.options.default_venv_backend = "uv|virtualenv"

SILENT_DEFAULT = True
SILENT_CODE_MODIFIERS = False

# Targets
CODE_LOCATIONS = ("src", "tests")
PYTHON_VERSIONS = ["3.13"]


@nox.session(python=PYTHON_VERSIONS, tags=["lint", "format"])
def ruff(session: nox.Session) -> None:
    """Run ruff over the sources and tests to enforce a consistent style.

    This session checks and optionally formats code in the src and tests
    directories so that contributors share a single, predictable style.
    The session parameter is the active nox session that controls the
    virtual environment and command execution for this invocation.
    It returns nothing and is intended for both local development and
    continuous integration runs.
    """
    # Determine the code locations to pass to ruff so that callers can override the defaults when desired.
    # Convert the default tuple into a list so that each path is passed as a separate argument to ruff.
    command_line_arguments = session.posargs or list(CODE_LOCATIONS)
    # Install ruff inside the session so that linting and formatting use a controlled version.
    session.install("ruff")
    # Run the ruff checks followed by an optional formatting pass to keep code style uniform.
    _run(session, "ruff", "check", *command_line_arguments)
    _run_code_modifier(session, "ruff", "format", *command_line_arguments)


@nox.session(python=PYTHON_VERSIONS, tags=["typecheck"])
def mypy(session: nox.Session) -> None:
    """Run mypy over src and tests to verify static type correctness.

    This session installs the project and type checking dependencies so
    that mypy can resolve imports and inspect type hints across src and
    tests. The session parameter is the active nox session that provides
    the environment used for installation and execution.
    It returns nothing and is designed for repeatable type checking in
    both local workflows and automated pipelines.
    """
    # Decide which paths mypy should analyse so that callers can narrow or broaden the default coverage.
    command_line_arguments = session.posargs or ("src", "tests")
    # Install the project and typing tools inside the session so that type information is available to mypy.
    session.install(".")
    # Install pytest because tests and fixtures are type-checked and rely on pytest symbols and decorators.
    session.install("mypy", "typing-extensions", "pytest")
    # Invoke mypy with the computed arguments so that static issues are reported consistently.
    _run(session, "mypy", *command_line_arguments)


@nox.session(python=PYTHON_VERSIONS, tags=["security"])
def bandit(session: nox.Session) -> None:
    """Run bandit over src and tests to detect common security issues.

    This session executes bandit against the main source and test code so
    that straightforward security problems can be highlighted early.
    The session parameter represents the active nox session that carries
    the environment configuration and command execution context.
    It returns nothing and is meant to provide a quick static security
    check in both interactive and automated runs.
    """
    # Compute the code locations to scan so that developers can refine the scope with explicit positional arguments.
    # Use a list derived from the default tuple so that bandit receives individual path arguments.
    command_line_arguments = session.posargs or list(CODE_LOCATIONS)
    # Install bandit into the session so that a consistent tool version is used across runs.
    session.install("bandit")
    # Run bandit with recursive scanning over the requested locations, skipping B101 because assert is used by pytest.
    _run(session, "bandit", "-r", "--skip", "B101", *command_line_arguments)


def _run(
    session: nox.Session,
    target: str,
    *command_line_arguments: str,
    silent: bool = SILENT_DEFAULT,
) -> None:
    """Run a command within a nox session using shared defaults.

    This helper wraps the raw session.run call so that common options
    such as the silent flag and external command allowance are applied
    consistently. The session parameter is the active nox session, the
    target parameter is the command to run, and command_line_arguments
    carries any additional arguments. The function returns nothing and
    keeps individual session bodies concise and readable.
    """
    # Delegate to session.run so that all commands use the same execution policy in terms of silence and external flags.
    session.run(target, *command_line_arguments, external=True, silent=silent)


def _run_code_modifier(
    session: nox.Session,
    target: str,
    *command_line_arguments: str,
) -> None:
    """Run a code-modifying command with code-specific silence settings.

    This helper is intended for tools that change files, such as format
    commands, so that their verbosity can differ from purely read-only
    checks. The session parameter is the active nox session, the target
    parameter is the command name, and command_line_arguments carries the
    remaining arguments. It returns nothing and delegates execution to
    the shared _run helper.
    """
    # Call the shared run helper with the modifier silence flag so that formatting tools can have tailored output behaviour.
    _run(session, target, *command_line_arguments, silent=SILENT_CODE_MODIFIERS)
