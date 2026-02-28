# Z I N O
A modular agent platform. ZINO stands for **Z**INO **I**s **N**ot **O**penclaw.


## On LLMs and Agents
Modern agent systems are a combination of a lot of moving parts. It is a necessary process, then, to identify and codify 
the nature of those parts, so we can come to a shared understanding of how a "simple" LLM can turn into a powerful
agent system.

On the LLM side, we can track the sequential invention of three different levels of sophistication, which we describe briefly below.
Complete explanations of our implementation of these components are given in the relevant sections:
- **The Chat-Instruct LLM Standard** is the standard post-training format for LLM products in use today.
      Chat mode is the name given to the conversation-like input format that contains "speakers" denoted
      as "USER" and "ASSISTANT". In Chat mode, LLMs generate text for the ASSISTANT side of the message
      pairs. Instruct mode is not conversational, but task-based; users make a request, often with
      accompanying data, and the model responds. This often includes tasks like translating, summarizing,
      extracting and formatting data, etc. The Chat-Instruct fine-tuning format is the foundation of
      all modern LLM products.  
- **Function Calling** parses inference output for specific tokens that denote calls to functions the
      application can execute programmatically, usually called "tools". In the Hermes style of function calling - the most common -
      inference is paused when a tool call block closes, and the application attempts to execute the function.
      The output of the execution is added to the already generated output, which is then fed back to the model
      as an incomplete "assistant" message. The model "sees" the beginning of an ASSISTANT response, the <tool call>
	  XML wraps, and the result inside of the <tool result> block, enabling it to continue generating with the
      newly added information. Function calling has become a standard for all the major LLM providers,
      and it enables the functioning of cross-chat memory, web browsing, image generation, and even
      complete agent platforms.
- **Agentic Runtimes** use skill definitions to chain together multiple tool calls, delegate jobs to
      sub-agents, create and respond to events, and perform other long-haul tasks. They build off of the
      same general idea behind function calling by adding additional depth and features to the way that
      interrupt tokens are handled. Agentic runtimes often mix in standard tools like *cron* (a linux tool
      used to schedule jobs), email, filesystem commands, etc., in order to accomplish more complex tasks.
      Skill definitions are similar to function definitions but contain a  longer set of specific instructions
      intended to guide the LLM through their task. As opposed to using strict tool signatures,
      skills usually start a new conversation "behind the scenes", where the skill data is injected into
      a clean context and used to perform multiple function-calls while the user-facing model waits and
      supervises.


If the above information seems like a lot to consider, don't be dissuaded: Agent runtimes really 
aren't that mysterious. Let's start with the end API call and work our way backwards through all
the parts that combine to build it. We're going to start with the strings - the prompt and 
conversation engineering, before we move on to the programmatic parts of a ZINO system.


## Building the LLM Request
### The API request
At the most basic level, we have the API call. This usually consists of four pieces of
data: the `URI`, `model path`, `API key`, and `data` objects. The first three are fairly
self-explanatory, consisting of the endpoint for the API we are calling, the specific
path (or name) to the model we want to use, and the API key that we need to access the
service. The data part is where it gets interesting.

Many API's have different endpoints and additional arguments that can be passed, but 
this is the bare mininum for most programs. Let's examine the `data` object, which contains
the list of messages that the model is meant to be continuing with its inference
generation.

### The message data
The message object in the standard openAI format is JSON with the following schema:
{
  [
    {
        "role" : "system",
        "content" : <SYSTEM_PROMPT>
    },
    {
        "role" : "user",
	"content" : "hello there jumoob"
    },
    {
        "role" : "assistant", 
        "content" : "My name is not jumoob"
    },
    {
        "role" : "user",
	"content" : "huh, well, would you like your name to be Jumoob?"
    }
  ]
}

This message object is further separated into two distinct pieces:
1. the `system prompt` opens the exchange, is usually optional, and contains the following data:
    - Personality
    - Tools and how to use them
    - Skills and how to run them
    - Soft Memory (Memory notes) - things to remember about the current project, the user, etc.

   The `system prompt` is central to the agent's proper functioning. A bad system prompt means 
   more tool failures and bad wide awareness.


2. The `chat history` is a sequence of messages between the `user` and the `assistant`. Agent programs compile
   this list from the following sources, typically in this order:
    - Task/Skill examples - These are real or fake user:asssistant exchanges which have been loaded from a
                            skill definition. This is commonly used in microAgents that have had a task delegated
                            to them, and which don't usually have their own chat context loaded.
    - Hard Memory - these are real or fake user:assistant exchanges that have been placed by a memory service
                    to help the model with understanding and fulfilling the user's request. It is common for these
                    to be supplied by a parallelized lookup in a vector memory store.
    - Conversation History - these are the actual previous exchanges between the user and model in 
                             the current context. These are often based on chat channels and may
                             saved and retrieved as unique "chats". This list of user:ass exchanges
                             is generally appended to with each completed turn in a chat session.
    - User Input - this is the actual string that the user is supplying as their next chat message.


Each seperate block represents a different micro service necessary for a complete agent platform to run,
but notice that each part is also completely optional, because this is all just a normal LLM inference call
once it's all put together.

That's it! We can fairly easily see that this gives us all the string data we need to run a successful
LLM inference with an agent. Once we have all of these in place, then it's up to the Function Calling
and Agentic Runtime to make the magic happen.


## Function Calling
The basic idea behind function calling is simple: we take the model's code, and we run it according to
the executable defined by the tool. So, if we want to make a bash tool, we tell the model what to generate
when it thinks bash would help, and it writes the code inside of the tool calling block.

However, there is a bit more to it than that. For starters, how do we stop the application from executing
dangerous or out-of-scope commands? How do we save tokens by cleaning and validating the code? What do we
do with the output, errors, etc.? These are the kinds of choices that can be isolated to the execution
component of the agent application.

The basic ZINO uses hermes-style function calling, which parses for <tool_call> tokens in the model's 
output and uses the structured inference text to run tools. Generation is paused at the tool 
invocation, the tool is executed, and a <tool_response> block containing the output is appended 
after the original call block. The model then sees the full call and response inline and 
continues generating. This preserves the original tool call tokens, which enables training on 
the full call-response-continuation sequence.

## Agentic Runtime
Agents are generally defined by their functionality built on top of the function calling. This includes the ability
to perform longer tasks by running sub-agents and longer and longer scripts. The former works quite well,
especially if we prompt the model to create and guide its temporary underlings well. The latter tends to
create problems which very quickly compound themselves as bugs take longer to hunt down.

Agents often use a combination of Skills and Events to do what they do. Skills are like nested tools - 
the agent creates a plan for its sub agent(s), and gets it to run tools until the agent is satisfied 
with the output. They'll often create a plan up front, and then execute it step by step, reasoning through
with themselves in the open. 



## The Complete Picture on ZINO

### Flow
A complete request follows this sequence:

A (message, channel_id) pair arrives at zino-daemon via Unix Domain Socket from a client. 
zino-daemon triggers zino-sys to supply the current system prompt (personality, tools, skills, 
soft memory). zino-daemon triggers zino-ctx to assemble the context history (skill/tool 
examples, hard memories, chat history, and the incoming user message). The fully assembled 
prompt is sent to zino-rtr, which manages the LLM API connection and returns the model's 
response. The response is passed through zino-agr, which parses it for interrupt tokens. If 
tool or skill calls are present, zino-agr coordinates with zino-exc to run them, handles 
retries, and loops until generation is complete. The final response is returned to zino-daemon, 
which passes it back to the client.

zino-sys and zino-ctx operate independently and at different frequencies — zino-sys only 
rebuilds when its inputs change (e.g. soft memory update, tool reload), while zino-ctx is 
rebuilt on every call.


### Parts
The current ZINO Agentic Runtime defines the following services available as separately 
packaged daemons on debian linux

- zino-daemon - This is the main entrypoint to the platform. It loads the main configuration, 
                contacts all the other services passing their boot configuration data as 
                needed, and provides a central control point for administration. All other 
                services depend on this guy to function, and it is one of the only required 
                modules, the other being zino-rtr. This guy does a lot of taking and 
                reformatting of data to be used by other components. It recieves all incoming 
                requests via Unix Domain Socket, and starts processing the (message, 
                channel_id) data. The channel_hash is optional, and default behaviour can be 
                set to a specific channel_id, silent drop, or to sessionless request. The 
                latter is the default. Inter-service communication uses Unix Domain Sockets in 
                the base implementation. The transport layer is an intentional upgrade path — 
                future versions may swap UDS for HTTPS, Tor, or other transports at the daemon 
                level without requiring changes to the other services.

- zino-rtr - This service is the first to receive a full prompt, and handles the connection 
             from the user to the LLM API. This is where you can optionally add retry 
             mechanics, API fallback routes, context limits, health checks, etc. -- On Load: 
             prepares LLM API connection (URL, Model, API Key)

- zino-agr - This is the the agentic runtime service. It parses the zino-router response for 
             interrupt tokens and interacts with the zino-exc service to manage tools, tasks, 
             sub-agents, and other specifications. -- On Load: Builds interrupt dictionary and 
             matches them to the tool calls/skill signatures. -- On Call: parses output 
             stream/whole, handles string manipulation for tool and skill use,
                routes input/output from execution, handles tool and skill retries, etc.

- zino-mem - This is the memory service, and it is responsible for storing and retrieving 
             session and channel-based chat histories and hard and soft memories. Chat 
             histories are straightforward, consisting of the real exchanges between the user 
             and the model. Hard memories are user:assistant exchanges pulled from memory by a 
             vector search. Soft memories are project wide and are usually set by a tool when 
             requested or deemed important, or by memory consolidation. All three types of 
             memory are optional, and overstuffing messages with memory can actually diminish 
             model capabilities. -- On Load:

- zino-sys - This is the system prompt service. It pulls together the data to fill out the 
             system prompt as defined above. This includes the personality details, tool 
             definitions, skill definitions, and soft memory notes. The system prompt only 
             updates when it is told to by other services and events, like when the memory 
             notes changes, or tools are reloaded. -- On Load: Builds the system prompt from a 
             four-part dictionary passed by the daemon with Personality, Tools,
                Skills, and Soft Memory pulled from their respective configurations.

- zino-ctx - This is the context history service. It puts together the context history for the 
             LLM API call. This includes tool examples; skills examples; hard memories (which 
             can change with each request); and chat history, if any. This means it is usually 
             updated with every call. -- On Load: -- On Call: Builds example exchanges from 
             tools, skills, and hard memories.
	
- zino-exc - This is the execution service. This component is responsible for executing the 
             code produced by tools and skills and enforcing secure access rules. It contains 
             the logic for the zino's executors - defined execution environments for running 
             different jobs. It can be extended to support events and to handle user input and 
             human-in-the-loop provided by the agent runtime. -- On Load: Checks for validators 
             (shellcheck) for its executors -- On Call: check perms, check code, execute code, 
             return all output to zino-agr


### Clients
Two clients are included as optional packages: zino-cli and zino-rest














