# STANDARDS
A ZINO compliant agent service will contain the following mandatory system components:
1. zino-daemon: This controls the Agentic logic including building and sending the 
                complete API call, handling interrupt tokens, io with the zino-exc for tool
                and function use, and streaming status and final data to the client.
                RESPONSE TO (message, ctxkey?) : Runs agent turn.
2. zino-ctx: Provides for the database connection, capable of storing all the plain-text
             data input that ZINO needs with keys as table names. Stores configuration,
             system prompts, memory, history, tools, and skills, and anything else needed.
             RESPONSE TO (message, ctxkey?) : Returns built message in openai 
             format (system, user, assistant,...)
3. zino-rtr: Handles the connection to and the specifics of the configured LLM API.
             This can include context length management, final string checks, and
             any retry/fallback logic.
             RESPONSE TO (message, ctxkey?) : Sends message to API provider, and 
             sends (streams if possible) the response back. ctxkey is treated as
             full override, and will pull the associated config in an attempt to
             use the API given. This can be used to make use of different models
             for different tasks, like using smaller models for basic summarizing.
4. zino-exc: Provides for the tool execution environment and management of sub-agents.
             The exc service can enforce execution rules, spawn and pipe sub-agents,
             and generally provide for the executive functions of the platform.
             RESPONSE TO (message, ctxkey?) : Expects to receive structured tool use.
             Will attempt to execute according to ctx config.  


Each component must accept and respond to a (message, ctxkey?) pair over UDS. This
response is what defines the service's role within a zino instance. All servers must
be able to handle a response of (message=None, ctxkey) from any other service without crashing.

These four system components can be used to explain the general flow of a request.
1. zino-daemon receives an incoming user request.
2. zino-daemon broadcasts the request to the three other services. zino-ctx responds 
   with the build message, while exc and rtr ready the correct configurations for the 
   ctxkey (they receive the ctxkey without any message (or None), and they check their
   cache-->zino-ctx for the loaded config).
3. zino-daemon sends the fully formed message to rtr.
4. rtr sends the message to the API.
5. rtr sends the message back to zino-daemon.
6. As the stream comes in, zino-daemon sends text to client, parses for interrupts
7. zino-daemon performs interrupt parsing, sending tools/skills as execution requests 
   to zino-exc, sending status updates to client.
8. zino-exc runs tools/skills and returns output to zino-daemon.
9. zino-daemon sends final output back to client.

