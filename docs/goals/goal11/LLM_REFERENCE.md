# Goal 11 external LLM reference panel

The LLM panel is external context, never a controlled comparison. The implementation freezes 100
proof and 100 SR IDs per configured model, retains raw responses and all terminal failure rows,
and reports claimed versus verifier-confirmed correctness separately. It makes no provider call
without credentials, exact provider model IDs, and explicit user spend confirmation.
