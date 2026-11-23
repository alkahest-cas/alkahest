"""Nonlinear corpus entries (``corpus_nonlinear``) that ``pdsolve`` solves and
verifies today: a lower bound (add an entry when it starts solving; one that
stops solving is a regression).  Entries that take more than a few seconds
(``zeldovich``, ``fitzhugh-nagumo``) are left out so a slow runner cannot
turn a budget refusal into a false regression."""

EXPECTED_SOLVED_M6 = {
    "abbasi-2001",
    "abbasi-2002",
    "abbasi-2003",
    "abbasi-2004",
    "abbasi-2005",
    "kdv",
    "kdv-1",
    "mkdv",
    "gardner",
    "burgers",
    "burgers-nu",
    "kdv-burgers",
    "fisher",
    "newell-whitehead",
    "allen-cahn",
    "huxley",
    "sine-gordon",
    "sine-gordon-wave",
    "phi4",
    "klein-gordon-cubic",
    "bbm",
    "boussinesq",
    "kawahara",
    "porous-medium",
}
