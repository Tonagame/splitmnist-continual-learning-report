# AI Work Log

This project used AI assistance as a development and documentation partner.

## How AI Was Used

AI help was used for:

- setting up the local Windows / Conda / CUDA workflow,
- interpreting the GMvandeVen repository,
- planning smoke tests and long experiments,
- writing PowerShell runners,
- debugging experiment failures,
- designing the LSR-lite prototype,
- checking protocol issues such as test-data leakage and Class-CL evaluation,
- creating CSV summaries and graphs,
- writing the GitHub Pages report,
- reorganizing the final GitHub repository.

## Important Human Decisions

The project direction and requirements were provided by the student:

- use Split MNIST,
- run on RTX 3070,
- compare Class-CL, Domain-CL, and Task-CL,
- add LSR-lite, Fourier, and ASW ablations,
- switch from using the original repository directly to implementing methods independently,
- keep limitations visible instead of hiding them.

## Transparency Note

The early phase used the GMvandeVen repository as a runnable reference. Later, because the assignment required implementing the methods ourselves, a clean-room implementation was added under:

`code/from_scratch/`

The documentation now separates:

- paper values,
- reference-code runs,
- our own implementation.

This separation is important for the defense.
