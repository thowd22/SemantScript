# SemantScript — Conversation Transcript

Source: ChatGPT thread "Fine Tune Backend Models". 16 turns, verbatim, chronological.

## Index

| # | User asks | Assistant's answer (key points) |
|---|-----------|--------------------------------|
| [1](#turn-1) | Has anyone fine-tuned small models for one job, e.g. a CRUD backend? | Yes — small models on narrow, verifiable tasks work. Fine-tune to implement a *backend specification*, not "write code"; eventually skip generating source code. |
| [2](#turn-2) | Clarifies: replace the backend itself with a fine-tuned LLM | Model-as-runtime. Keep data out of the weights (DB stays), model handles business logic; consequences and the hidden CS problem (correctness/state). |
| [3](#turn-3) | Is this what Jev (TypeSafe) does? What does "type-safe" mean? | Jev = unstructured state → typed probabilistic decisions, not free-form strings. Close to our idea, but our idea goes further (whole backend). |
| [4](#turn-4) | How did they make it type-safe? Can we combine the concepts? | Jev doesn't generate JSON; outputs are constrained to types. Combine with backend idea → "ISA for neural software", tiny deterministic kernel, uncertainty as a first-class signal, training/experiment ideas. |
| [5](#turn-5) | Could we build a rapid training framework: describe backend in English → generate backend? | English → formal behavior spec → auto-generated training set → fast training. Runtime stays boring; deployment ideas; prototype plan. |
| [6](#turn-6) | With a Jev-like architecture, could we reach deterministic-software accuracy? | Type-safety makes the *interface* deterministic, not the *decision*. Three layers of correctness; verifiers we control; "stronger version" and product hiding inside. |
| [7](#turn-7) | Missing pieces: syntax and parallelism. Why is Jev fast? Controllers with natural-language functions mixed with code? | Jev removes autoregressive generation from the hot path. Controller syntax proposal, compiler-discovered parallelism, training reframed, deterministic code as leverage — "this is a language, not a backend framework". |
| [8](#turn-8) | Is anyone else doing this? | Closest: "Compile by Training" paper (PAW, Sept 2026). Overlaps and gaps vs. our idea; landscape of related projects. |
| [9](#turn-9) | Don't like PAW; let's design our own | First-principles design in 10 points: TypeScript host, `neural<T>` primitive (not a prompt), non-autoregressive model, one model/many functions, compiler parallelism, inference stages, training syntax, deterministic assertions, controllers, tiny first model. Four project components. |
| [10](#turn-10) | Neural functions need explicit inputs/outputs to/from variables | Syntax like a normal typed function; strongly typed structured inputs and outputs, multiple outputs, variable flow between neural and deterministic code, a **confidence** concept; beginnings of a language spec. |
| [11](#turn-11) | Mixing with TypeScript means this isn't backend-only — we can use libraries | TypeScript as systems layer; library interoperability; frontend use; Node advantages; libraries feeding and becoming neural functions; cleaner syntax proposal. |
| [12](#turn-12) | Need an md file with investigation and decision points | Handed off to Work mode to produce the design/investigation doc. |
| [13](#turn-13) | Need a name | Avoid AI/LLM/backend in the name. Candidates: Synapse, Semant, Nerve, Cortex, Sema. |
| [14](#turn-14) | What about SemantScript? | Endorsed: JavaScript → TypeScript → SemantScript. Syntax examples. |
| [15](#turn-15) | Need a full unedited transcript | Handed off to Work mode to create this file. |
| [16](#turn-16) | Transcript export instruction (verbatim, `semantscript-conversation-transcript.md`) | — |

---

<a id="turn-1"></a>
## User

Has anyone tried fine tuning small models to do one job? Let’s take building a crud app backend. Basically fine tune a backend instead of writing code.

## Assistant

Yes. What you’re describing is a real and increasingly interesting direction: **instead of making a general model good at programming, make a small model extremely good at one constrained software-generation task.**

There are already strong signals that this works. Recent work has shown substantial improvements from LoRA/QLoRA on sub-3B models for code generation, and other research is explicitly training models around narrow, verifiable programming tasks. :chatgpt-content-reference{index="0"}

But I’d take your idea one step further.

### Don't fine-tune it to "write backend code"

Fine-tune it to implement a **backend specification**.

Imagine a 1–3B model whose universe is:

```text
Input:

Create a Todo API.

Todo:
  id: uuid
  title: string
  completed: boolean
  created_at: datetime

Endpoints:
  POST   /todos
  GET    /todos
  GET    /todos/{id}
  PUT    /todos/{id}
  DELETE /todos/{id}

Database: postgres
Auth: JWT
```

And its output is *always* something like:

```text
/app
  models.py
  schemas.py
  routes/todos.py
  services/todos.py
  database.py
  auth.py
  main.py
/tests
  test_todos.py
Dockerfile
pyproject.toml
```

You could constrain the entire training distribution to one stack:

**FastAPI + Pydantic + SQLAlchemy + PostgreSQL + Alembic + pytest**

Now the model doesn't need to understand React, Kubernetes, Rust, Java, random Python libraries, etc. It essentially becomes a **natural-language-to-backend compiler**.

That's where small models become interesting.

### And there is evidence for exactly this strategy

A particularly relevant example is Learn2Zinc: small 1B/3B models are fine-tuned specifically to translate natural-language optimization problems into executable MiniZinc programs. :chatgpt-content-reference{index="1"}

Likewise, people are training small code models with LoRA on individual codebases. A very recent project called CodeFinetuner, for example, fine-tunes Qwen2.5-Coder-3B against a particular repository and evaluates exact match, CodeBLEU, edit similarity, etc. :chatgpt-content-reference{index="2"}

And Code2LoRA is exploring an even more interesting variant: generating **repository-specific LoRA adapters** so that a model effectively learns a codebase rather than receiving the entire repo in its context window. :chatgpt-content-reference{index="3"}

So your intuition is pretty well aligned with where some current research is heading.

## The part I think could make your CRUD idea work unusually well

CRUD generation has something most AI coding tasks don't:

**correctness is automatically measurable.**

You can spin up the generated application and run:

```text
Does it compile?
       ↓
Does it start?
       ↓
Does migration succeed?
       ↓
POST /users → 201?
       ↓
GET /users/{id} → correct object?
       ↓
PUT → object changed?
       ↓
DELETE → object gone?
       ↓
Invalid input → 422?
       ↓
Unauthorized → 401?
```

That means your dataset doesn't necessarily need humans rating outputs.

You have a **verification oracle**.

That's important because recent work using reinforcement learning with verification feedback found that even 0.6B and 1B models can improve code-generation correctness when unit tests/static analysis provide the reward signal. :chatgpt-content-reference{index="4"}

So I'd probably build the training pipeline as:

```text
        Synthetic API specification
                  │
                  ▼
          Teacher model
       generates backend
                  │
                  ▼
       Build Docker image
                  │
                  ▼
          Run test suite
                  │
          ┌───────┴────────┐
          │                │
        FAIL             PASS
          │                │
       discard             ▼
                   training example
                           │
                           ▼
                    SFT / QLoRA
                           │
                           ▼
                     Small model
                           │
                           ▼
                       RLVR
                           │
                  tests = reward
```

You could generate **hundreds of thousands of synthetic CRUD applications** fairly cheaply.

Vary relationships, validation rules, pagination, uniqueness, indexes, auth, RBAC, soft deletion, transactions, filtering, error handling, etc.

Then the model learns the mapping:

> specification → known-good application

rather than merely memorizing Python syntax.

## Eventually you might not even have it generate source code

This is where I think the idea gets more interesting.

Have the model generate an intermediate representation:

```yaml
entities:
  User:
    fields:
      id: uuid
      email:
        type: string
        unique: true
      name: string

  Post:
    fields:
      id: uuid
      title: string
      body: text
      author:
        relation: User

routes:
  User:
    - create
    - list
    - get
    - update
    - delete

auth:
  type: jwt

permissions:
  Post:
    create: authenticated
    update: owner
    delete: owner
```

Then a **deterministic compiler** turns that into FastAPI/SQLAlchemy/Alembic code.

Now the model only needs to solve:

```text
English
   ↓
structured backend representation
   ↓
deterministic compiler
   ↓
actual backend
```

That is *much* easier for a 1–3B model than reliably emitting 5,000 tokens of internally consistent Python.

And your model can become tiny.

Potentially:

```text
General coding model
30B–100B+
      ↓

Specialized backend model
3B
      ↓

Highly constrained backend-spec model
0.5B–1.5B
```

The interesting experiment would be finding **how small you can go before reliability collapses**.

A 500M–1B model that takes:

> "Build an inventory API. Products belong to categories, SKUs are unique, managers can modify inventory, users can read it, support pagination."

and produces a formally validated backend specification in ~100 ms would be much more useful than it initially sounds.

It also connects nicely to your earlier idea about local security models: **a collection of small specialists rather than one enormous general model**. A local system could route tasks to specialized adapters/models—backend generation, Terraform analysis, log classification, IAM-policy analysis, vulnerability triage, and so on.

For this CRUD experiment, I'd start around **1.5–3B parameters + QLoRA + synthetic verified training data**, rather than training from scratch. The key innovation wouldn't really be the model—it would be the **automatic data generation + execution/test verification loop**. That's the piece that could turn a surprisingly small model into something extremely reliable.

<a id="turn-2"></a>
## User

That’s what I was considering, replacing the backend with a fine tuned llm

## Assistant

Yes — and that’s a substantially different idea from what I described.

You mean **the model itself is the backend runtime**, not that it generates a backend.

Instead of:

```text
Client
  ↓
REST API
  ↓
Business logic
  ↓
ORM
  ↓
Database
```

you'd have something closer to:

```text
Client
  ↓
Small fine-tuned model
  ↓
persistent state
```

For example, train it on interactions like:

```text
REQUEST:
POST /users
{"name":"Bob","email":"bob@example.com"}

STATE:
[...]

RESPONSE:
201
{"id":"123","name":"Bob","email":"bob@example.com"}

STATE_MUTATION:
INSERT users ...
```

The interesting question then becomes: **how much of the conventional backend can the model actually absorb?**

### I wouldn't put the data inside the weights

That's the crucial distinction I'd make.

Fine-tuning could teach the model the application's **behavior**, but I'd keep mutable state external:

```text
                 ┌─────────────────────┐
request ────────►│  Fine-tuned 0.5-3B │
                 │       model         │
                 │                     │
                 │ "the application"   │
                 └──────────┬──────────┘
                            │
                     structured actions
                            │
                            ▼
                 ┌─────────────────────┐
                 │   State primitive   │
                 │                     │
                 │ get / put / query   │
                 └─────────────────────┘
```

So instead of implementing:

```python
@app.post("/users")
def create_user(...):
    validate(...)
    check_permissions(...)
    user = User(...)
    db.add(user)
    db.commit()
    return user
```

the model has learned:

> what creating a user means.

It receives the request and relevant state and emits something constrained like:

```json
{
  "status": 201,
  "mutations": [
    {
      "op": "insert",
      "collection": "users",
      "value": {
        "id": "$uuid",
        "name": "Bob",
        "email": "bob@example.com"
      }
    }
  ]
}
```

A tiny deterministic runtime validates and executes that.

So you've replaced perhaps **95% of the application backend** with a model, while keeping the dangerous pieces deterministic.

### This gets really interesting with business logic

CRUD itself isn't where I'd expect the biggest payoff.

Imagine a traditional backend has 100k lines containing rules like:

```text
premium users can...
admins can...
orders over $500 require...
refunds within 30 days...
unless the account...
except when...
```

Instead of translating all of those requirements into application code, you fine-tune the model directly on the desired **input → state transition → output behavior**.

Then the architecture starts looking almost absurdly small:

```text
HTTP/gRPC
    │
    ▼
┌──────────────┐
│  App Model   │  ← business logic
│    ~1B       │
└──────┬───────┘
       │
       ├── read_state()
       ├── query_state()
       ├── mutate_state()
       └── emit_response()
              │
              ▼
          Postgres
```

Your "backend framework" might literally be a few thousand lines responsible for inference, authentication context, DB transactions, schema validation and model-tool execution.

Everything application-specific lives in the weights.

### There are some fascinating consequences

Deploying application logic becomes:

```text
app-v17.gguf
```

instead of:

```text
Git repo
Docker image
dependencies
framework
ORM
controllers
services
business logic
...
```

And changing behavior becomes training:

```text
Before:

refund(order)
→ denied after 30 days


New examples:

VIP customers may receive refunds
within 60 days


Fine tune / adapter
        ↓

app-v18.gguf
```

You could even imagine:

```text
base-backend-model
       │
       ├── ecommerce.lora
       ├── inventory.lora
       ├── ticketing.lora
       └── crm.lora
```

where the base model learns HTTP semantics, transactions, authorization concepts, CRUD, validation, etc., while a LoRA contains the application's behavior.

### But there's a nasty computer-science problem hiding here

A conventional program gives you:

```text
same input + same state
        ↓
same output
```

An LLM doesn't inherently guarantee that.

And backend systems really care about invariants:

```text
balance >= 0

email UNIQUE

inventory >= 0

user cannot modify
another user's record

transaction either
fully commits or rolls back
```

I would therefore **not allow the model to directly mutate storage**.

Make it propose state transitions:

```text
LLM
 ↓
proposed transaction
 ↓
schema validator
 ↓
authorization engine
 ↓
invariant checker
 ↓
transaction engine
 ↓
DB
```

That creates an interesting hybrid:

> **The neural network determines application behavior; a tiny deterministic kernel guarantees correctness.**

It's almost analogous to an operating system: the model runs in "user space," while the backend kernel controls privileged operations.

And *that* version of your idea strikes me as much more interesting than "LLM writes a CRUD app."

You'd essentially be asking whether **application software can be represented primarily as learned weights instead of source code**.

A CRUD application would be a fantastic experiment because you could build a reference backend, generate millions of randomized state/request combinations, train perhaps a **0.5–1.5B model**, and then run differential tests:

```text
Traditional backend ──┐
                      ├── compare outputs/state
Neural backend ───────┘

1,000,000 test cases
```

If the neural backend reached something like **99.99% behavioral equivalence**, then the failures become the really interesting research question: *what kinds of software behavior don't compress well into weights?*

That would be an experiment I'd genuinely want to see.

<a id="turn-3"></a>
## User

Is this especially what the new Jev ai does? Is that what they mean bye being typsafe?

## Assistant

Yes — **Jev is very close to the architectural direction we were just discussing**, but with an important distinction: it isn't trying to replace an entire backend. It's trying to replace the **decision/business-logic portions** of software with a model.

TypeSafe describes Jev as essentially:

> **unstructured program state → typed probabilistic decisions**

rather than an LLM that generates arbitrary strings. :chatgpt-content-reference{index="0"}

### What they mean by "type-safe"

A normal LLM might do this:

```text
State:
Customer requested a refund 43 days after purchase.

Prompt:
Should we refund them?

LLM:
"Based on the company's policies, I think you
should probably deny the refund because..."
```

Your application now has to interpret that text.

Structured-output mode improves this:

```json
{
  "decision": "deny",
  "confidence": 0.94
}
```

But it's still fundamentally a language model generating tokens that are subsequently constrained/parsed.

Jev takes a different approach. You define the possible outputs **before inference**. TypeSafe currently exposes three decision primitives: Choice, Score, and Noul (roughly a Boolean/probability). :chatgpt-content-reference{index="1"}

Conceptually:

```typescript
decision = jev({
    state: order,

    question: {
        type: "choice",
        options: [
            "approve",
            "deny",
            "manual_review"
        ]
    }
})
```

The model's output space is therefore:

```text
approve
deny
manual_review
```

It **cannot return**:

```text
"Well, after carefully considering..."
```

or

```text
"approove"
```

or some malformed JSON object.

That's what TypeSafe means by type-safe: the model returns a value belonging to the predefined output type rather than arbitrary generated text. Their announcement explicitly says the possible outputs and structure are defined in advance and claims the model therefore doesn't make type errors. :chatgpt-content-reference{index="2"}

Importantly, **type-safe doesn't mean correct**. Jev can choose a valid but wrong option. TypeSafe addresses that separately by returning probabilities/confidence and training with what it calls Reinforcement Learning for Calibrated Decisions (RLCD). :chatgpt-content-reference{index="3"}

---

## And this is surprisingly close to your "LLM backend" idea

Consider a traditional backend:

```text
HTTP Request
     │
     ▼
Controller
     │
     ▼
Service
     │
     ├── if customer.age > ...
     ├── if order.status == ...
     ├── if refund.days < ...
     └── if account.type == ...
     │
     ▼
Database
```

Jev is targeting something more like:

```text
HTTP Request
     │
     ▼
Application state
     │
     ▼
┌───────────────┐
│     JEV       │
│               │
│   Decision    │
│    Model      │
└───────┬───────┘
        │
        │ typed decision
        ▼
  approve_refund
        │
        ▼
 deterministic code
        │
        ▼
     Database
```

That's why TypeSafe calls it **"machine-native intelligence."** Their argument is essentially that conventional LLMs are optimized to produce words for humans, whereas Jev produces decisions intended to be consumed directly by programs. :chatgpt-content-reference{index="4"}

And they explicitly describe Jev as something like a **"frontier-intelligence function call."** :chatgpt-content-reference{index="5"}

---

### Where your idea goes further

Jev appears to be saying:

```text
Code
 │
 ├── HTTP
 ├── DB
 ├── transactions
 ├── orchestration
 │
 └── Jev
      │
      └── intelligent decisions
```

You're proposing something closer to:

```text
         HTTP
          │
          ▼
 ┌──────────────────┐
 │ Fine-tuned Model │
 │                  │
 │ Application      │
 │ Behavior         │
 └────────┬─────────┘
          │
          ▼
  deterministic kernel
          │
 ┌────────┼─────────┐
 │        │         │
 ▼        ▼         ▼
get()   put()    transaction()
 │        │         │
 └────────┼─────────┘
          ▼
       Postgres
```

That's a considerably more radical proposition.

Jev answers:

> **"Given this state, what decision should the software make?"**

Your model would answer:

> **"Given this state and request, what state transition should the application perform?"**

And there's a really elegant way to combine Jev's type-safety idea with yours.

Instead of letting your model generate arbitrary SQL:

```sql
DELETE FROM users ...
```

define a tiny algebra of legal operations:

```typescript
type BackendAction =
    | Get<Entity>
    | Query<Entity>
    | Insert<Entity>
    | Update<Entity>
    | Delete<Entity>
    | Reject<Reason>
    | Respond<Response>
```

Then the model's entire universe is **typed state transitions**.

For example:

```text
POST /users

{
  "email": "tyler@example.com"
}
```

could result internally in:

```text
Insert<User>(
    email = "tyler@example.com"
)
```

The runtime—not the model—performs the SQL transaction.

That would give you something very much in the spirit of Jev:

**request + state → typed decision**

except the decision type is rich enough to express application behavior.

And I think that's the key insight connecting what you were thinking about to TypeSafe's work: **the model doesn't necessarily need to generate code or language at all.**

It can *be* the application's learned logic while communicating with the deterministic world exclusively through a small, type-safe instruction set.

That's a much more technically plausible version of "replace the backend with a neural network" than asking an LLM to generate arbitrary SQL/JSON on every request.

<a id="turn-4"></a>
## User

How did they make it typesafe? Could we combine the two concepts?

## Assistant

Yes. And after looking more closely at what TypeSafe has published, I think **combining their idea with your "model-as-backend" idea is probably the right architecture.**

There is one important caveat: TypeSafe has **not publicly disclosed enough of Jev's internal architecture to reproduce exactly how they implement the typed output heads**. They say it's a "new model architecture," uses a parallel sampler, and is trained with Reinforcement Learning for Calibrated Decisions (RLCD), but they haven't published the architecture details. :chatgpt-content-reference{index="0"}

What they *have* disclosed tells us enough to build our own version of the concept.

## Jev isn't generating JSON

This is the biggest distinction.

With a normal LLM:

```text
Transformer
    ↓
token logits
    ↓
"{" 
    ↓
"action"
    ↓
":"
    ↓
"insert"
    ↓
...
```

Even constrained decoding is still fundamentally generating tokens.

Jev instead exposes a fixed decision space. TypeSafe currently defines three primitives:

```text
Noul
   P(true) = 0.97

Choice<A,B,C>
   P(A) = .02
   P(B) = .96
   P(C) = .02

Score<0..4>
   probabilities over levels
```

For a `Choice`, for example, you supply the allowed choices and their semantic descriptions. Jev returns **one of those keys plus a probability distribution over the allowed choices**. It cannot invent a fourth choice. :chatgpt-content-reference{index="1"}

So conceptually it's much closer to a learned classifier:

```text
             ┌─────────────┐
state ──────►│ Transformer │
             └──────┬──────┘
                    │
                 embedding
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       INSERT     UPDATE     DELETE
        .91        .06        .03
```

than:

```text
LLM → "I think you should INSERT..."
```

That's where the type safety comes from.

The model isn't being *asked* to obey the type.

**The output space itself represents the type.**

---

# Now combine that with your backend idea

This is where I think there's a legitimately interesting architecture.

Don't make our fine-tuned model produce:

```json
{
  "operation": "insert",
  "table": "users",
  "values": {
    "name": "Bob"
  }
}
```

Instead define an **instruction set for backend applications.**

Something like:

```text
BackendOp<T>

READ<T>
QUERY<T>
CREATE<T>
UPDATE<T>
DELETE<T>
RESPOND<T>
REJECT<Error>
```

Our model selects among those operations.

Imagine the application schema is:

```typescript
interface User {
    id: UUID
    email: Email
    name: string
    role: Role
}

type Role =
    | "user"
    | "admin";
```

And the request is:

```text
POST /users

{
    "email": "bob@example.com",
    "name": "Bob"
}
```

The neural backend evaluates:

```text
Application state
+
request
+
authenticated principal
+
schema
```

and produces:

```text
BackendAction<User>

CREATE     .997
UPDATE     .001
DELETE     .000
READ       .002
```

That output is intrinsically type constrained.

---

# But we can go considerably further

The next decision can itself be typed.

After:

```text
CREATE<User>
```

the model needs to populate a `User`.

Instead of generating JSON, ask typed questions:

```text
email:
    COPY<request.email>       .9998
    GENERATE                  .0001
    NULL                      .0001

name:
    COPY<request.name>        .998

role:
    USER                      .997
    ADMIN                     .003
```

Now we're effectively executing a **neural program**.

```text
                     Request
                        │
                        ▼
                ┌──────────────┐
                │ Neural App   │
                └──────┬───────┘
                       │
                typed decisions
                       │
              ┌────────┴────────┐
              ▼                 ▼
         CREATE<User>        REJECT
              │
              ▼
        typed User fields
              │
              ▼
      ┌──────────────────┐
      │ Deterministic    │
      │ Backend Kernel   │
      └────────┬─────────┘
               │
          validation
          authorization
          invariants
          transaction
               │
               ▼
            Postgres
```

The model **never sees SQL**.

It can't say:

```sql
DROP TABLE users
```

because `DROP TABLE` literally doesn't exist in its action space.

That's a much stronger property than prompting an LLM:

> "Never drop a table."

---

# This starts looking like an ISA for neural software

This is the part I'd be most interested in exploring.

Think about CPUs.

A CPU doesn't understand:

```python
refund_customer(order)
```

It understands a tiny instruction set:

```text
LOAD
STORE
ADD
CMP
JMP
...
```

We could define the equivalent for learned backend software:

```text
GET<T>

FIND<T>

CREATE<T>

UPDATE<T>

DELETE<T>

CHECK<Predicate>

CHOOSE<T>

CALL<Service>

EMIT<Event>

RESPOND<T>

FAIL<Error>
```

Call it something like:

**Neural Backend ISA**

Then the model doesn't generate source code.

It executes this instruction set probabilistically.

---

## The deterministic kernel becomes tiny

The trusted portion of the backend might only need to implement:

```text
┌─────────────────────────────┐
│ HTTP / gRPC                 │
├─────────────────────────────┤
│ Authentication              │
├─────────────────────────────┤
│ Schema/type validation      │
├─────────────────────────────┤
│ Authorization boundaries    │
├─────────────────────────────┤
│ Invariant enforcement       │
├─────────────────────────────┤
│ Transaction manager         │
├─────────────────────────────┤
│ Neural instruction executor │
├─────────────────────────────┤
│ Postgres                    │
└─────────────────────────────┘
```

Everything that would normally be:

```text
controllers/
services/
handlers/
business_logic/
repositories/
```

could potentially disappear into:

```text
application.safetensors
```

---

# And Jev gives us another crucial ingredient: uncertainty

Jev doesn't just return:

```text
DELETE<User>
```

It returns the probability distribution and confidence. TypeSafe explicitly designed the API around this idea. :chatgpt-content-reference{index="2"}

That means our runtime can have a rule like:

```python
if confidence > .999:
    execute(action)

elif confidence > .95:
    verify(action)

else:
    fallback()
```

For dangerous operations:

```text
READ
threshold = .95

CREATE
threshold = .99

UPDATE
threshold = .999

DELETE
threshold = .99999
```

And the fallback could be conventional code, a larger reasoning model, or human approval depending on the application.

That gives us something conventional software doesn't normally have:

**the application knows when its learned business logic is uncertain.**

---

# Training becomes particularly interesting

This is where your original CRUD idea is perfect.

Generate random schemas:

```text
Users
Orders
Products
Inventory
Tickets
Invoices
Subscriptions
Organizations
Permissions
...
```

Generate business rules:

```text
Only admins can delete users.

Users can modify their own profile.

Orders cannot be modified after shipping.

Refunds require the order to be <30 days old.

Inventory cannot fall below zero.
```

Then generate millions of situations:

```text
STATE
REQUEST
EXPECTED ACTION
EXPECTED MUTATION
EXPECTED RESPONSE
```

Example:

```text
STATE

User:
  id: 12
  role: normal

Request:

DELETE /users/27
```

Expected:

```text
Action:

REJECT<Unauthorized>
```

And because we control the simulator, **the correct answer is automatically known**.

That makes training data essentially unlimited.

---

# Then the really wild experiment

Train progressively smaller models.

```text
7B
 ↓
3B
 ↓
1B
 ↓
500M
 ↓
250M
 ↓
100M
```

and measure:

```text
behavioral accuracy
type violations
invariant violations
latency
memory
requests/sec
```

The Jev approach suggests that abandoning general text generation may allow surprisingly small models to remain useful. TypeSafe says Jev evaluates its typed questions in parallel rather than autoregressively generating output, which is central to its claimed efficiency. :chatgpt-content-reference{index="3"}

So the research question becomes:

> **How many parameters does it actually take to represent the business logic of a typical application when the model doesn't have to speak English or write code?**

My suspicion is that it's **dramatically fewer than a coding LLM**.

You could conceivably end up with:

```text
        Traditional SaaS
             │
       200k LOC backend
             │
             ▼
        Neural SaaS
             │
     300M parameter model
             +
       5k LOC kernel
             │
             ▼
          Postgres
```

And that would be much more profound than "AI writes software."

It would mean **the weights *are* the software**.

The first thing I'd build to test the thesis isn't even a complicated app. I'd create a conventional CRUD reference implementation, define a tiny typed backend ISA, generate maybe 1–10 million randomized request/state/action tuples, fine-tune a small open model, and differential-test the neural implementation against the conventional one. That would tell us very quickly whether the basic premise works.

<a id="turn-5"></a>
## User

Ok, this is getting interesting. Could we develop a rapid training framework where we could describe backend behavior in human language and use that to generate our backend?

## Assistant

Yes. I think the strongest version is **not** “fine-tune directly on the English description.” It’s a compiler/training pipeline where English becomes a formal behavioral specification, that specification generates executable training cases, and a small model is trained to implement the behavior through a typed backend ISA.

The developer experience could eventually be almost this simple:

```text
Build an order-management backend.

Users belong to organizations.
Users can create and view orders for their organization.
Only managers can cancel orders.
Orders can't be changed after they ship.
Products have inventory.
Creating an order reserves inventory.
Inventory can never go below zero.
Managers can refund an order within 30 days.
```

Then:

```bash
neural-backend build app.md
neural-backend test
neural-backend serve
```

## The architecture

I'd separate the system into **five layers**:

```text
             Human specification
                     │
                     ▼
         ┌──────────────────────┐
         │ Spec Compiler        │
         │ LLM used at build    │
         │ time only            │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ Formal App Spec      │
         │                      │
         │ entities             │
         │ relationships        │
         │ permissions          │
         │ invariants           │
         │ transitions          │
         └──────────┬───────────┘
                    │
             ┌──────┴──────┐
             ▼             ▼
      Test Generator   Data Generator
             │             │
             └──────┬──────┘
                    ▼
              Teacher/Oracle
                    │
                    ▼
           millions of examples
                    │
                    ▼
              rapid fine-tune
                    │
                    ▼
          ┌────────────────────┐
          │ Neural Application │
          │ 100M? 500M? 1B?   │
          └──────────┬─────────┘
                     │
              typed Backend ISA
                     │
                     ▼
          deterministic runtime
                     │
                     ▼
                  Postgres
```

The critical component is the **Formal App Spec**.

### English → formal behavior

Your description might compile into something like:

```yaml
entities:

  User:
    id: uuid
    organization: Organization
    role:
      enum: [user, manager]

  Order:
    id: uuid
    organization: Organization
    status:
      enum: [pending, shipped, cancelled]
    created_at: datetime

  Product:
    id: uuid
    inventory: integer

permissions:

  Order.read:
    allow:
      user.organization == order.organization

  Order.cancel:
    allow:
      user.role == manager
    require:
      order.status != shipped

invariants:

  - Product.inventory >= 0

transitions:

  create_order:
    effects:
      - create Order
      - decrement Product.inventory

  refund_order:
    require:
      - user.role == manager
      - age(order) <= 30 days
```

At this stage **we can statically analyze the application**.

For example:

```text
✓ All referenced entities exist
✓ All fields type check
✓ All transitions terminate
✓ All mutations have authorization rules
✓ Inventory invariant defined
⚠ DELETE Product has no policy
```

That formal intermediate representation becomes extremely important.

---

## Then generate the training set automatically

For every rule, create adversarial combinations.

Take:

```text
Managers may refund orders within 30 days.
```

The generator produces:

```text
manager + 10 days  → REFUND
manager + 29 days  → REFUND
manager + 30 days  → REFUND
manager + 31 days  → DENY

user + 10 days     → DENY
user + 29 days     → DENY
user + 31 days     → DENY

manager + cancelled order → ?
manager + nonexistent order → ?
manager + another organization → ?
```

Notice something important:

**the training framework discovers ambiguities in the human specification.**

Instead of silently guessing, the compiler can stop:

> You said managers can refund orders within 30 days.
>
> Can managers refund orders belonging to another organization?
>
> **A)** No, only their organization  
> **B)** Yes, any organization  
> **C)** Custom rule

That's a huge advantage over normal AI code generation.

The framework could progressively interrogate the developer until the behavioral state space is sufficiently specified.

---

# Training could potentially be extremely fast

We don't need to teach the model:

```text
English
Python
SQL
HTTP
FastAPI
Postgres syntax
JSON generation
software architecture
```

The base model already understands the generic neural-backend ISA.

We're only teaching:

**this application's decision boundaries.**

I'd actually envision two components:

```text
         Base Neural Runtime Model
                ~500M?
                   │
                   │ frozen
                   ▼
             ┌───────────┐
             │ App LoRA  │
             │           │
             │ 5–100 MB? │
             └───────────┘
```

The base model understands:

```text
GET
FIND
CREATE
UPDATE
DELETE
CHECK
CALL
RESPOND
DENY
```

and generic concepts like users, permissions, relationships, dates, quantities, transactions, etc.

The LoRA teaches it:

```text
YOUR APPLICATION
```

That raises the possibility that compiling a new backend isn't a six-hour training job.

Eventually it might be:

```text
$ neural build ecommerce.md

Compiling specification...
Generating scenarios........ 42,381
Generating adversarial....... 18,922
Training adapter.............. 47 sec
Validating.................... 12 sec

Tests
──────────────────────────────
Behavioral        100.000%
Authorization     100.000%
Invariants        100.000%
Adversarial        99.997%

Artifact:

ecommerce.napp
18.4 MB
```

I don't know whether **47 seconds** or an **18 MB adapter** is realistic—we'd have to experimentally determine that—but that's the product target I'd pursue.

---

# There's an even better trick

We don't necessarily need a giant teacher model for most training examples.

Because the formal spec is executable.

Once:

```text
English
   ↓
Formal Spec
```

has happened, a deterministic simulator can generate:

```text
state + request → correct action
```

Millions of times.

So expensive LLM inference occurs primarily during **compilation**.

Training data generation becomes CPU work.

```text
                 GPT / Claude / local 70B
                         │
                         │ once
                         ▼
English ──────────► Formal Specification
                         │
                         ▼
                  deterministic
                    simulator
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
         example      example      example
            │            │            │
            └────────────┼────────────┘
                         ▼
                  10M examples
                         │
                         ▼
                     LoRA train
```

That makes the economics much better.

---

# The runtime should remain deliberately boring

I wouldn't neuralize everything.

Keep these deterministic:

```text
TLS
HTTP parsing
authentication/identity
database driver
transactions
schema validation
referential integrity
hard security boundaries
invariant enforcement
logging/auditing
```

Let the model replace:

```text
controllers
service layer
business rules
workflow logic
conditional logic
policy decisions
```

So:

```text
┌───────────────────────────────┐
│        Neural Backend         │
│                               │
│   learned application logic   │
├───────────────────────────────┤
│     Deterministic Kernel      │
│                               │
│ auth │ types │ invariants     │
│ txns │ audit │ persistence    │
├───────────────────────────────┤
│           Postgres            │
└───────────────────────────────┘
```

That's a system I'd be much more comfortable deploying.

---

## And deployment becomes fascinating

Instead of:

```text
Docker image
  1.2 GB

Python/Node runtime
framework
dependencies
application code
ORM
controllers
services
```

you could eventually deploy:

```text
neural-runtime
       +
schema.nb
       +
orders.lora
```

Potentially one generic runtime hosts **thousands of different applications** merely by swapping adapters.

```text
                  Neural Runtime
                        │
          ┌─────────────┼──────────────┐
          ▼             ▼              ▼
      CRM LoRA     Inventory LoRA   Orders LoRA
          │             │              │
          ▼             ▼              ▼
       CRM DB       Inventory DB     Orders DB
```

That has some pretty wild implications for serverless computing.

---

# I would build the first prototype differently than you'd expect

**Don't start by training a model.**

Build the language and runtime first.

Version 0:

```text
English
   ↓
LLM
   ↓
Formal Backend Spec
   ↓
Deterministic interpreter
   ↓
Postgres
```

If we can't accurately express real applications through the formal spec/ISA, putting a neural model in the middle won't help.

Once that works:

**Version 1**

```text
Formal spec → generated test universe
```

**Version 2**

```text
Tests → fine-tuned 1B model
```

**Version 3**

```text
Traditional interpreter
         vs
Neural implementation

differential fuzz testing
```

And then we answer the fundamental question experimentally:

> **Can a relatively small neural network learn an application's complete behavioral specification with reliability approaching deterministic software?**

If yes, *then* we'd have something considerably more interesting than another AI code generator. We'd have a compiler where the output artifact isn't source code—**it's a trained model.**

<a id="turn-6"></a>
## User

But with an architecture similar to Jev, we could reach a level the level of accuracy that deterministic software.

## Assistant

Potentially, but there's an important distinction:

**A Jev-like architecture can make the *interface* deterministic/type-safe. It does not make the model's *decision* deterministic or always correct.**

TypeSafe itself explicitly says Jev can get things wrong; its customer agreement warns outputs can be inaccurate. What they claim is mathematically guaranteed is **schema/type correctness**—not semantic correctness. :chatgpt-content-reference{index="0"}

That said, I think your backend idea has an advantage Jev doesn't generally have: **we control the domain, the specification, the state machine, and the verifier.** That could let us get much closer to deterministic-software reliability.

### Think of correctness as three layers

Suppose our application has:

```typescript
cancelOrder(user, order)
```

The neural model could output only:

```text
ALLOW   0.999997
DENY    0.000003
```

There's no possibility of malformed JSON, an invented action, SQL injection, `DROP TABLE`, etc. That's the Jev-like part. TypeSafe describes Jev similarly: predefined typed outputs, produced in parallel with probabilities and confidence. :chatgpt-content-reference{index="1"}

But it could still choose `ALLOW` when the correct answer is `DENY`.

So I wouldn't rely on a magical accuracy number like 99.9999%. I'd architect the system so **a neural mistake does not necessarily become a software correctness failure**.

```text
                    Human requirements
                           │
                           ▼
                   Formal specification
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      Neural application           Hard constraints
             │                           │
             │ typed decision            │ deterministic
             ▼                           │
      ┌───────────────┐                  │
      │ CREATE<Order> │                  │
      │ confidence    │                  │
      │ 0.999998      │                  │
      └───────┬───────┘                  │
              │                           │
              └────────────┬──────────────┘
                           ▼
                 Verification Kernel
                           │
                  ┌────────┴────────┐
                  │                 │
                VALID            INVALID
                  │                 │
                  ▼                 ▼
               execute           reject
                  │
                  ▼
               Postgres
```

Now we have two very different classes of rules.

**Hard invariants** remain deterministic:

```text
inventory >= 0

email is unique

foreign keys valid

money transferred:
    debit == credit

organization boundaries

transaction atomicity

schema constraints
```

Those should *never* depend on model confidence.

The neural model handles behavioral rules:

```text
Should we approve this refund?

Which workflow should this request enter?

Which account should this object belong to?

Does this request satisfy the cancellation policy?

What operation is the user requesting?
```

That's actually consistent with TypeSafe's published workflow philosophy: decompose tasks into narrow intelligent judgments and use ordinary code for things that can be computed deterministically. Their eval description explicitly says they defer to code where possible. :chatgpt-content-reference{index="2"}

### But our framework can go beyond that

Because we're compiling a **specific application**, we can automatically test the learned model against the formal specification.

Imagine training finishes and we run:

```text
Training complete.

Generated scenarios:       12,492,183
Property tests:             4,201,482
Adversarial scenarios:        892,441
Boundary conditions:           83,210

Type violations:                    0
Invariant violations:               0
Incorrect decisions:               17
Low-confidence incorrect:          17
High-confidence incorrect:          0
```

That last distinction is huge.

If all 17 failures occur below our confidence threshold:

```text
confidence < 0.999
        │
        ▼
deterministic fallback
```

then the deployed behavior on that test universe can still be **100% correct**, despite the neural model not being 100% accurate.

This is where TypeSafe's emphasis on calibrated confidence becomes particularly relevant. Jev is explicitly designed to expose probabilities/confidence so ordinary software can decide when to act and when to escalate. :chatgpt-content-reference{index="3"}

## There's an even stronger version

We could make the **formal specification executable**.

Then every neural decision carries a proof obligation:

```text
Model proposes:

UPDATE<Order> {
    status: CANCELLED
}

        ↓

Verifier asks:

✓ Order exists?
✓ User authenticated?
✓ Same organization?
✓ User is manager?
✓ Current state != SHIPPED?
✓ Transition PENDING → CANCELLED allowed?
✓ All resulting invariants hold?

        ↓

COMMIT
```

At that point, the neural network isn't really trusted.

It's proposing computation.

The deterministic kernel decides whether that computation is legal.

That resembles ideas from proof-carrying code, capability systems, database constraints and typed intermediate representations much more than today's agent architectures.

### Which changes our goal

We don't actually need:

> **model accuracy = deterministic software accuracy**

We need:

> **system correctness = deterministic software correctness**

Those are very different objectives.

Our model could theoretically only be 99.9% accurate while the overall backend approaches conventional deterministic reliability because uncertain or invalid transitions never make it through the kernel.

And unlike a general model, we can hammer this particular application with generated tests before deploying it.

---

### I think there's a product hiding here

What we're describing now isn't really an "AI backend framework."

It's closer to a **neural application compiler**:

```text
$ nb create

Describe your application:

> Users create projects.
> Projects contain tasks.
> Members can modify tasks.
> Only project owners can delete projects.
> Completed projects are immutable.
> ...

             ↓

      Requirements model
             ↓

      Formal specification
             ↓

     ambiguity detection
             ↓

        developer Q&A
             ↓

     executable reference
             ↓

     synthetic scenarios
             ↓

      train App Adapter
             ↓

    exhaustive/fuzz/property
          verification
             ↓

       signed artifact


        project.nba
```

And the deployed artifact might contain:

```text
project.nba

├── schema
├── invariants
├── permissions
├── transition graph
├── app adapter weights
├── confidence thresholds
├── verification policies
└── test/eval certificate
```

The runtime would be generic.

**The artifact is the application.**

That's the part I think is genuinely novel compared with Jev: TypeSafe is building a general primitive for making probabilistic typed decisions inside conventional software. :chatgpt-content-reference{index="4"} **We'd use a similar primitive as the execution substrate for an entire learned application, with a deterministic verification kernel surrounding it.**

And that suggests a very concrete next step: before worrying about model architecture, we should design the **typed Neural Backend ISA + formal specification language**. If we get those two pieces right, we could initially run the exact same system with a normal model, then progressively replace it with a specialized small model.

<a id="turn-7"></a>
## User

The only missing pieces are syntax and parallel processing. Jev is super fast, how is it so fast? On the syntax side or defining the training. Seems like we could organize into controllers that act like api endpoints for front end calls. Then there could be functions defined within the controllers that are natural language. These natural language functions could be intermingled with traditional programming primitives.

## Assistant

Yes. I think you just described the programming model that would make this practical.

And Jev's newly published details clarify why it's fast: **they removed autoregressive string generation from the hot path.** TypeSafe says Jev evaluates outputs in parallel, with 70–500 ms end-to-end latency, and reports 40–200× speedups for comparable System-One-shaped queries. It also says Jev uses a new model architecture and a “parallel sampler,” but has not published enough architectural detail to reproduce that sampler. :chatgpt-content-reference{index="0"}

The really important observation for us is that **parallelism exists at two levels**.

### Why Jev can be so fast

A conventional LLM answering 15 questions effectively does something like:

```text
tokens → transformer → token
                   ↓
             transformer → token
                         ↓
                   transformer → token
                              ...
```

Even structured JSON is still a sequence of tokens.

Jev instead takes state plus a collection of predefined questions and produces the decisions in parallel:

```text
                         ┌── isFraud:       0.002
                         ├── isRefundable:  0.991
STATE ──► MODEL ─────────┼── isVIP:         0.871
                         ├── risk:          2.1
                         └── action:
                             REFUND         .93
                             REVIEW         .06
                             DENY           .01
```

TypeSafe explicitly says it produces “all outputs in a single query” rather than generating one token at a time. Their real workflows exploit this heavily: the customer-service example starts with **eleven independent readings of the conversation at once**, then only evaluates dependent follow-ups afterward. :chatgpt-content-reference{index="1"}

That's almost exactly what we'd want.

---

# Your controller idea solves the syntax problem

I wouldn't invent an entirely new programming language initially.

I'd make this a library for an existing typed language—probably **TypeScript first**.

Imagine:

```typescript
@controller("/orders")
class OrdersController {

    @post("/")
    async create(
        user: User,
        request: CreateOrder
    ): Promise<Order> {

        const allowed = decide<boolean>(`
            Determine whether this user is allowed to
            create this order.

            Users may only order for their own organization.
            Suspended users cannot place orders.
        `);

        if (!allowed)
            throw Forbidden();

        const inventory = await db.products.get(
            request.productId
        );

        require(inventory.quantity >= request.quantity);

        return db.orders.create({
            userId: user.id,
            productId: request.productId,
            quantity: request.quantity
        });
    }
}
```

Notice what's happened.

We have **normal programming**:

```typescript
if (!allowed)
    throw Forbidden();

require(inventory.quantity >= request.quantity);

return db.orders.create(...)
```

intermixed with **learned programming**:

```typescript
const allowed = decide<boolean>(`
    Determine whether this user is allowed...
`);
```

That is very close to TypeSafe's philosophy. Their eval harnesses explicitly separate things that should be calculated in ordinary code—dates, sums, statuses—from judgments that benefit from a model. :chatgpt-content-reference{index="2"}

---

# But I would change the syntax slightly

The natural-language function needs an explicit type.

Something like:

```typescript
const allowed = neural<boolean>(`
    A user may create an order when they are ordering
    for their own organization and their account is active.
`);
```

Or:

```typescript
const action = neural<
    "approve" | "deny" | "review"
>(`
    Decide how this refund should be handled.

    Refunds within 30 days are normally approved.
    Managers can override the normal policy.
    Suspicious refund activity must be reviewed.
`);
```

Or:

```typescript
const risk = neural<Score<0, 10>>(`
    Evaluate the fraud risk of this order.
`);
```

Now we've essentially recreated Jev's three concepts:

```text
boolean              → Noul-like
union/enum           → Choice-like
Score<0,10>           → Score-like
```

but embedded them directly into a programming language.

---

# Now add controllers

This becomes really elegant.

```typescript
@controller("/refunds")
class RefundController {

    @post("/:id")
    async refund(ctx: Context): Promise<RefundResult> {

        const order = await db.orders.get(ctx.params.id);

        require(order.exists);
        require(ctx.user.organizationId === order.organizationId);

        const [
            eligibility,
            fraudRisk,
            customerIntent
        ] = neural.parallel({

            eligibility: neural<boolean>(`
                Determine whether this order qualifies
                for a refund under our refund policy.
            `),

            fraudRisk: neural<Score<0, 5>>(`
                Determine the likelihood that this
                refund request is fraudulent.
            `),

            customerIntent: neural<
                "refund" |
                "replacement" |
                "support"
            >(`
                Determine what outcome the customer
                actually wants.
            `)

        });

        if (fraudRisk >= 4)
            return review();

        if (!eligibility)
            return deny();

        if (customerIntent === "replacement")
            return replace(order);

        return refund(order);
    }
}
```

**This is where Jev-style parallel execution becomes extremely powerful.**

Those three questions don't depend upon one another.

So the compiler creates:

```text
                     ┌── eligibility
                     │
state ──► MODEL ─────┼── fraudRisk
                     │
                     └── customerIntent
```

**one inference.**

Not three.

---

# And the compiler can discover parallelism automatically

We shouldn't even require `neural.parallel()` eventually.

Consider:

```typescript
const intent = neural<Intent>(`
    Determine the customer's intent.
`);

const risk = neural<Risk>(`
    Determine this customer's fraud risk.
`);

const sentiment = neural<Sentiment>(`
    Determine customer sentiment.
`);

if (risk === "high")
    return review();

if (intent === "refund" && sentiment === "angry")
    ...
```

The compiler constructs the dependency graph:

```text
                  REQUEST
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       intent       risk     sentiment
          │          │          │
          └──────────┼──────────┘
                     ▼
                application
                   logic
```

Therefore:

```text
Inference round #1

[intent, risk, sentiment]

            ↓

Inference round #2

questions depending on round #1

            ↓

Inference round #3
...
```

That is effectively **AI instruction-level parallelism**.

TypeSafe's published workflows already use this general pattern: independent questions run together, then conditional questions are evaluated only when prior results require them. :chatgpt-content-reference{index="3"}

---

# Now the training problem gets really interesting

Here's where I would diverge slightly from our previous idea.

The source code itself becomes the training specification.

Consider:

```typescript
const refundable = neural<boolean>(`
    Customers may normally return physical products
    within 30 days.

    Digital purchases cannot be returned after they
    have been downloaded.

    Enterprise customers have a 60 day return window.
`);
```

Our compiler knows:

```text
INPUT TYPE
Context + Customer + Order

OUTPUT TYPE
boolean

BEHAVIOR
natural-language specification
```

That gives us everything needed to generate training data.

During:

```bash
neural build
```

we could compile every neural function.

```text
src/
  orders.controller.ts
  refunds.controller.ts
  users.controller.ts

             │
             ▼

       Neural Compiler

             │

    ┌────────┼────────┐
    ▼        ▼        ▼

 refundable risk   intent

    │        │        │
    └────────┼────────┘
             ▼

      generate cases
             ↓
      validate cases
             ↓
       train adapter
             ↓

        app.nmodel
```

The developer never creates a training dataset.

**The program is the training dataset specification.**

---

# And traditional code becomes incredibly valuable

This is the part I like most about your proposal.

We don't have to decide:

> neural software **or** traditional software?

Instead, programmers decide exactly where ambiguity belongs.

```typescript
// Deterministic
const days = now.diff(order.createdAt).days;

// Neural
const legitimate = neural<boolean>(`
    Does the customer's explanation appear
    consistent with a legitimate return?
`);

// Deterministic
if (days > 30)
    return DENIED;

// Neural
const resolution = neural<Resolution>(`
    Given the customer's circumstances,
    determine the most appropriate resolution.
`);

// Deterministic
switch (resolution) {
    case "refund":
        return refund(order);

    case "credit":
        return credit(order);

    case "review":
        return escalate(order);
}
```

This is extremely similar to what TypeSafe found in their workflow evaluations: **structured workflows combining code with narrow model judgments performed better than asking models to solve the entire policy in one prompt.** :chatgpt-content-reference{index="4"}

---

## And I think we should steal another idea from compilers

Treat each natural-language expression as an actual **typed instruction** in the intermediate representation.

Source:

```typescript
const risk = neural<Risk>(`
    Assess this transaction for fraud.
`);
```

Compiler IR:

```text
NEURAL_CALL #47

inputs:
    transaction: Transaction
    customer: Customer

output:
    Risk {
        LOW,
        MEDIUM,
        HIGH
    }

policy_hash:
    84fa3...

dependencies:
    none
```

Then the optimizer sees:

```text
CALL #47 ─┐
CALL #48 ─┼── independent
CALL #49 ─┘
```

and transforms them into:

```text
PARALLEL_NEURAL_CALL {
    #47,
    #48,
    #49
}
```

Very much like a compiler performing instruction scheduling.

---

# Which means we're not actually designing a backend framework anymore

We're starting to describe a **mixed deterministic/neural programming language**.

And the abstraction boundary is unusually clean:

```text
Traditional code
──────────────────────────────
Exact computation

Math
Dates
Database operations
Transactions
Loops
State
Cryptography
Hard security invariants


Natural-language functions
──────────────────────────────
Judgment

Classification
Interpretation
Policy
Fuzzy rules
Intent
Semantic relationships
Complex business decisions
```

Both are **typed**.

Both participate in the same control flow.

Both can exist inside the same function.

The compiler automatically extracts the neural dependency graph, batches independent decisions, generates training examples, fine-tunes the application model, and packages the resulting weights alongside the conventional executable.

So your deployed controller might *look* like this to the programmer:

```typescript
@post("/refund")
async refund(ctx: Context) {

    const order = await db.order(ctx.orderId);

    require(order.customerId === ctx.user.id);

    const eligible = neural<boolean>(`
        Determine whether this customer's circumstances
        qualify for a refund under our policy.
    `);

    if (!eligible)
        return deny();

    await db.refund(order);

    return success();
}
```

but what actually gets deployed is:

```text
                  compiled application

             ┌────────────────────┐
HTTP ───────►│ deterministic code │
             └─────────┬──────────┘
                       │
                neural instruction
                       │
                       ▼
             ┌────────────────────┐
             │ App Model / LoRA   │
             │                    │
             │ parallel decision  │
             └─────────┬──────────┘
                       │
                  typed result
                       │
                       ▼
             ┌────────────────────┐
             │ deterministic code │
             │                    │
             │ transactions       │
             │ invariants         │
             │ database           │
             └────────────────────┘
```

**That feels like the right prototype.**

And importantly, we don't need to reproduce Jev's proprietary architecture to test the programming model. We could build v0 with an existing small open model plus classification heads/batched inference, prove out the controller syntax, compiler IR, dependency scheduler and training pipeline, and *then* investigate a non-autoregressive model architecture optimized specifically for these typed neural instructions. Jev's public results suggest that eliminating string generation and structuring workflows around parallel decisions is exactly where a large portion of the latency advantage comes from. :chatgpt-content-reference{index="5"}

<a id="turn-8"></a>
## User

Is anyone else doing this?

## Assistant

Yes — and I found something especially relevant that changes my answer from “this seems novel” to **“there is now a research project extremely close to the core idea.”**

The closest thing I found is a September 2026 paper called **“Compile by Training: Turning Natural-Language Specifications into Local Neural Functions.”** It builds on a system called **Program-as-Weights (PAW)**. Their exact premise is: describe a function in natural language, use teacher models at compile time to synthesize examples, train a small adapter for a fixed neural interpreter, and then package that adapter as a reusable function that runs locally without the teacher. :chatgpt-content-reference{index="0"}

Their example is essentially:

```python
fn = paw.compile_and_load(
    "Classify urgent emails"
)

fn("Need this today")
# → "urgent"
```

The resulting `.paw` artifact is effectively **a function implemented as learned weights rather than conventional source code**. Their public SDK literally describes itself as “compile natural language specs into tiny neural functions that run locally.” :chatgpt-content-reference{index="1"}

That's remarkably close to what we've been discussing.

### Where PAW overlaps with our idea

Their pipeline is:

```text
Natural-language function
          │
          ▼
   Teacher models
          │
          ▼
Synthetic examples
          │
          ▼
Fine-tune small interpreter
          │
          ▼
     function.paw
          │
          ▼
   local inference
```

Their newer “compile by training” implementation first initializes a neural program, asks teacher models for task-specific input/output examples, fine-tunes the local interpreter, and packages the result. :chatgpt-content-reference{index="2"}

That's almost exactly our:

```text
neural<boolean>(`
    Determine whether this order
    qualifies for a refund.
`)
```

becoming a trained function.

And their reported compile time is already on the order of **a minute**, rather than hours, for the higher-accuracy training approach. They report 83.6% semantic accuracy on their FuzzyBench-Hard subset. :chatgpt-content-reference{index="3"}

So the “rapid neural compilation” part is not hypothetical anymore.

---

### But they're not building the backend architecture we're describing

This is the important distinction.

PAW is primarily aimed at **fuzzy text functions**:

```text
classify
extract
normalize
repair
fuzzy-match
triage
route
```

Their tooling explicitly describes those as its target use cases. :chatgpt-content-reference{index="4"}

We're proposing something broader:

```text
@Controller("/orders")
class Orders {

    @Post("/")
    create(req: CreateOrder) {

        require(req.quantity > 0);

        const allowed = neural<boolean>(`
            Determine whether this customer
            may place this order according
            to our purchasing policy.
        `);

        if (!allowed)
            throw Forbidden();

        return db.orders.create(req);
    }
}
```

with the **compiler** turning `neural<boolean>` into a specialized learned function.

That's different because conventional and learned computation participate in the same typed program.

---

### There are a few other projects circling the idea

**Eulyph** is an experimental AI-native language that explicitly separates a “fluid” AI track from a “rigid” deterministic track. Natural-language inference is first-class syntax, while math, functions, I/O, JSON and serving APIs remain deterministic. That's surprisingly close to your suggestion of intermingling natural-language functions with conventional primitives. :chatgpt-content-reference{index="5"}

**IntentLang** similarly experiments with making natural-language intent an executable construct embedded directly into Python rather than merely using prompts/function calling. :chatgpt-content-reference{index="6"}

**Prajna** is another interesting research prototype because it lets semantic functions start as LLM calls and then attempts to lower them into cheaper implementations—rules, distilled statistical heads, etc.—provided they satisfy an accuracy contract. :chatgpt-content-reference{index="7"}

And **Linkly** takes a different approach: developers declare goals/business rules, an LLM lowers those into a semantic IR, and conventional compilation eventually produces native machine code. So it's AI-assisted semantic compilation rather than the business logic remaining neural at runtime. :chatgpt-content-reference{index="8"}

### The landscape looks roughly like this

```text
                        Natural language
                              │
       ┌──────────────────────┼─────────────────────┐
       │                      │                     │
       ▼                      ▼                     ▼

     Linkly                 PAW/Jev             Eulyph/
                                                 IntentLang

 Natural language        learned function       natural language
       ↓                       ↓                executes through
 semantic IR                weights                  LLM
       ↓                       ↓
 machine code              inference


                             OUR IDEA
                                │
                                ▼

                 typed hybrid programming model

             ┌──────────────────┴─────────────────┐
             │                                    │
     deterministic code                    neural code

 require(x > 0)                   neural<RefundDecision>(`
 db.get(...)                         Apply our refund policy
 transaction(...)                    to this request.
 crypto(...)                      `)
             │                                    │
             └──────────────────┬─────────────────┘
                                ▼
                         dependency compiler
                                ▼
                       parallel neural ISA
                                ▼
                         application model
```

And **that last combination is the gap I haven't found a mature project filling**.

More specifically, I haven't found a system combining all of these properties:

1. Controllers/endpoints as normal application structure.
2. Natural-language functions as first-class typed expressions.
3. Conventional code and neural expressions freely intermixed.
4. Natural-language functions **compiled into specialized weights**, rather than prompting a large LLM at runtime.
5. A Jev-like finite/typed output space rather than arbitrary text generation.
6. Compiler analysis of dependencies so independent neural functions execute in **one parallel inference round**.
7. A deterministic kernel controlling persistence, transactions, auth and hard invariants.
8. The resulting thing being treated as an **application/backend programming model**, rather than a library for classification.

There are projects doing **pieces of every one of those**. PAW in particular validates a huge chunk of the premise. But I haven't found one putting those pieces together into the backend language/runtime we've been sketching.

And PAW actually gives us a shortcut. We wouldn't necessarily have to begin by inventing our own neural compiler. Their code is public and MIT-licensed, and their compile-by-training implementation is public too. :chatgpt-content-reference{index="9"}

[ProgramAsWeights GitHub](https://github.com/programasweights/programasweights-python?utm_source=chatgpt.com)

[Compile-by-Training implementation](https://github.com/programasweights/compile-by-training?utm_source=chatgpt.com)

So a very practical first prototype could be **TypeScript/Python frontend → typed `neural<T>` functions → PAW-style compilation → deterministic controller runtime**. That would let us test the programming model before spending months developing a Jev-like parallel model architecture.

<a id="turn-9"></a>
## User

I don’t like paw after looking into it. I’d like to design our own.

## Assistant

I agree. I’d treat the other systems as evidence that pieces are feasible, but **design ours from first principles around the programming model we want**.

I’d also narrow the initial thesis. We do not need to solve “neural backends” all at once. We need to prove one primitive:

> **A typed natural-language function can be compiled into a small, fast neural function whose output is constrained by its type and which can execute alongside ordinary code.**

Everything else follows from that.

## 1. Start with the language

I like TypeScript as the host language because its type system gives us a natural interface.

I would make the primitive look approximately like:

```typescript
const decision = neural<"approve" | "deny" | "review">({
    inputs: { user, order },

    behavior: `
        Determine whether this order can be refunded.

        Customers can normally receive refunds within 30 days.
        Enterprise customers have 60 days.
        Fraudulent requests should be sent for review.
    `
});
```

And normal code remains normal:

```typescript
@Post("/orders/:id/refund")
async refund(ctx: Context) {

    const order = await db.orders.get(ctx.params.id);

    // deterministic
    require(order.customerId === ctx.user.id);

    // learned
    const decision = neural<"approve" | "deny" | "review">({
        inputs: {
            user: ctx.user,
            order
        },

        behavior: `
            Determine whether this order can be refunded.

            Customers normally have 30 days.
            Enterprise customers have 60 days.
            Suspicious circumstances require review.
        `
    });

    // deterministic again
    switch (decision) {

        case "approve":
            return db.transaction(() =>
                refund(order)
            );

        case "review":
            return createReview(order);

        case "deny":
            return denied();
    }
}
```

That alone would be a pretty compelling programming model.

## 2. `neural<T>` isn't a prompt

This distinction should be fundamental.

The programmer isn't saying:

> Send this text to an LLM.

They're declaring:

```text
f(inputs) → T
```

with the natural language defining the behavior of `f`.

At build time:

```text
neural<T>
    │
    ▼
Neural Compiler
    │
    ├── analyze behavior
    ├── analyze input types
    ├── analyze output type
    ├── generate training cases
    ├── generate adversarial cases
    ├── train
    ├── calibrate
    └── verify
            │
            ▼
        NeuralFunction
```

At runtime, the English ideally isn't even necessary.

The compiled artifact might simply contain:

```text
function_id: refund_decision_7af31

input_schema:
    User
    Order

output:
    categorical[3]

weights:
    ...

calibration:
    ...

verification:
    ...
```

That's very different from prompt engineering.

## 3. I wouldn't start with an autoregressive LLM

This is probably the most consequential design choice.

If the function returns:

```typescript
"approve" | "deny" | "review"
```

why would we generate tokens?

We want something more like:

```text
                    Encoder
                       │
User ────────────────► │
Order ───────────────► │
                       │
                       ▼
                 hidden state
                       │
                       ▼
                classification
                     head
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          approve     deny     review
           .973       .012      .015
```

**One forward pass.**

No:

```text
token → token → token → token
```

No JSON decoding.

No constrained generation.

No parsing.

That gets us much closer to the fundamental reason a Jev-like system can be fast.

## 4. But one model should support many functions

We don't want:

```text
refund-model.bin
fraud-model.bin
intent-model.bin
permission-model.bin
```

Instead:

```text
             Application Encoder
                     │
          ┌──────────┼──────────┐
          │          │          │
          ▼          ▼          ▼
       refund       fraud      intent
        head         head       head
```

Potentially:

```text
Base encoder
   100M–1B parameters

Application adapter
   small

Function heads
   tiny
```

Now a controller might contain 50 neural functions while sharing almost all computation.

That becomes critical for performance.

---

# 5. Parallelism should be a compiler feature

Suppose we write:

```typescript
const fraud = neural<FraudRisk>(...);

const intent = neural<CustomerIntent>(...);

const sentiment = neural<Sentiment>(...);

const refundable = neural<boolean>(...);
```

The compiler sees:

```text
fraud       inputs=[user,order,request]
intent      inputs=[request]
sentiment   inputs=[request]
refundable  inputs=[user,order]
```

No dependencies.

So instead of four inference calls:

```text
encoder → fraud

encoder → intent

encoder → sentiment

encoder → refundable
```

we do:

```text
                         ┌── FraudHead
                         │
                         ├── IntentHead
INPUT ──► ENCODER ───────┼── SentimentHead
                         │
                         └── RefundHead
```

**One encoder pass.**

That's potentially enormous.

The compiler automatically constructs the neural execution DAG.

---

# 6. Dependent neural functions become inference stages

Suppose:

```typescript
const intent = neural<Intent>(...);

const resolution = neural<Resolution>({
    inputs: {
        intent,
        customer,
        conversation
    },
    behavior: `...`
});
```

Now the compiler knows:

```text
             request
                │
                ▼

          inference stage 1

              intent
                │
                ▼

          inference stage 2

            resolution
```

A larger controller might compile into:

```text
                     REQUEST
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
          fraud       intent      sentiment
            │           │
            └─────┬─────┘
                  ▼
              resolution
                  │
                  ▼
                action
```

That's a neural execution plan.

Very similar conceptually to what a database does when it creates a query plan.

---

# 7. Training syntax needs one additional feature

Natural language alone isn't enough.

Developers need to specify **examples** when behavior is subtle.

I'd support:

```typescript
const refund = neural<RefundDecision>({

    inputs: { user, order, request },

    behavior: `
        Determine whether this refund should
        be approved, denied or reviewed.

        Normal refunds are allowed within 30 days.
        Enterprise customers have 60 days.
        Suspicious requests require review.
    `,

    examples: [
        {
            input: enterpriseOrder(45),
            output: "approve"
        },
        {
            input: normalOrder(45),
            output: "deny"
        }
    ]
});
```

Now we've got three sources of truth:

```text
Types
  +
Natural-language specification
  +
Examples
```

The compiler can synthesize training data from all three.

---

# 8. Add deterministic assertions to the specification

This could become particularly powerful:

```typescript
const decision = neural<RefundDecision>({
    inputs: { user, order },

    behavior: `
        Apply our refund policy.
    `,

    constraints: [
        never(
            ({ order }) =>
                order.status === "fraudulent",
            "approve"
        ),

        always(
            ({ order }) =>
                order.ageDays > 90,
            "deny"
        )
    ]
});
```

Now:

**natural language handles fuzzy behavior.**

**code defines absolute behavior.**

And training can specifically generate cases around those boundaries.

---

# 9. Controllers become a natural abstraction

Eventually I'd like application code to look almost boring:

```typescript
@Controller("/claims")
class ClaimsController {

    @Post("/")
    async submit(ctx: Context) {

        const claim =
            await db.claims.create(ctx.body);

        const [
            category,
            fraud,
            urgency
        ] = neural({

            category: {
                type: ClaimCategory,
                behavior: `
                    Categorize this insurance claim.
                `
            },

            fraud: {
                type: FraudRisk,
                behavior: `
                    Assess the likelihood of fraud.
                `
            },

            urgency: {
                type: Urgency,
                behavior: `
                    Determine how urgently this
                    claim requires attention.
                `
            }
        });

        require(fraud !== "critical");

        return routeClaim({
            claim,
            category,
            fraud,
            urgency
        });
    }
}
```

The programmer doesn't think about models.

They think:

```text
code when exact
natural language when semantic
```

That's the abstraction.

---

# 10. The first model should be intentionally tiny

I wouldn't start with 1B.

I'd actually start around **100–300M parameters**.

We aren't trying to make it know everything.

We want to answer:

> Can an encoder trained specifically for typed semantic functions learn a collection of application behaviors?

Our first benchmark could contain perhaps:

```text
100 applications

1,000 neural functions

10–20 output types

10M generated examples
```

Measure:

```text
Accuracy
Calibration error
P50/P95 latency
Throughput
Memory
Batch scaling
Parallel-head scaling
Training time
Adapter size
```

Then compare:

```text
Our 200M model

vs

1B generative model

vs

7B generative model

vs

traditional LLM structured output
```

If the 200M model gets comparable decision accuracy at dramatically lower latency, **then we know the architecture is worth pursuing.**

---

## The project really has four components

I'd structure the repository something like:

```text
neural-lang/
│
├── compiler/
│   ├── types
│   ├── parser
│   ├── neural-ir
│   ├── dependency-graph
│   └── optimizer
│
├── trainer/
│   ├── synthetic-data
│   ├── adversarial-data
│   ├── distillation
│   ├── calibration
│   └── evaluation
│
├── model/
│   ├── encoder
│   ├── adapters
│   ├── heads
│   └── parallel-runtime
│
└── runtime/
    ├── http
    ├── controller
    ├── persistence
    ├── transactions
    ├── verifier
    └── inference
```

And I would **not build the backend framework first**.

I'd build one thing:

```typescript
const result = neural<
    "approve" | "deny" | "review"
>({
    inputs,
    behavior: `...`
});
```

Compile that to our own tiny model/head, train it, and get a benchmark showing **sub-10-ms local inference with extremely high classification accuracy** on a narrowly defined task.

If that works, parallel functions are next. Controllers come third.

At that point we'd have proven the underlying primitive rather than building a framework around an assumption.

And there's one architectural question I'd investigate very early: whether **every neural function really needs separately trained weights**, or whether we can train a universal semantic encoder once and have the compiler generate extremely small function-specific heads/adapters. If the latter works, “compilation” could eventually take seconds instead of minutes. That would make the programming model much more compelling.

<a id="turn-10"></a>
## User

I’d also like to determine inputs and outputs to the ai function. We should be able to consume and output to variables.

## Assistant

Exactly. I think that should be a **core language property**, not something inferred magically from whatever happens to be in the prompt.

A neural function should behave like an ordinary typed function:

```text
(inputs) → neural computation → (outputs)
```

The difference is simply that its implementation is specified in natural language and compiled into learned computation.

## I think the syntax should look like a normal function

Instead of:

```typescript
const result = neural<Decision>({
    inputs: { user, order },
    behavior: `...`
});
```

I'd prefer something closer to:

```typescript
neural function refundDecision(
    user: User,
    order: Order,
    request: RefundRequest
): RefundDecision {

    """
    Determine how this refund request should be handled.

    Customers normally have 30 days to return an order.
    Enterprise customers have 60 days.
    Suspicious requests should be reviewed.
    """
}
```

Then use it exactly like conventional code:

```typescript
const decision = refundDecision(
    ctx.user,
    order,
    ctx.body
);

if (decision === "approve") {
    await refund(order);
}
```

That's a much stronger abstraction.

From the programmer's perspective:

```typescript
function add(a: number, b: number): number {
    return a + b;
}

neural function classify(
    customer: Customer,
    message: string
): Intent {
    """
    Determine what the customer is trying to accomplish.
    """
}
```

Both are functions.

One is implemented with deterministic instructions.

One is implemented with learned weights.

---

## Structured inputs become especially important

We're not limited to text.

Suppose:

```typescript
interface Order {
    id: UUID;
    total: Money;
    createdAt: Date;
    status: OrderStatus;
    items: OrderItem[];
}

interface Customer {
    id: UUID;
    tier: CustomerTier;
    accountAge: Duration;
    previousRefunds: number;
}
```

Then:

```typescript
neural function evaluateRefund(
    customer: Customer,
    order: Order,
    explanation: string
): RefundDecision {

    """
    Evaluate this refund request according to our policy.

    Consider the customer's history, order circumstances
    and explanation.

    Suspicious circumstances should be reviewed rather
    than automatically denied.
    """
}
```

The compiler knows exactly what information enters the learned computation.

That's useful for training, security, reproducibility and optimization.

It also means this:

```typescript
evaluateRefund(customer, order, explanation)
```

does **not** secretly have access to:

```text
database
environment variables
other users
HTTP headers
filesystem
network
global state
```

unless explicitly passed in.

That's a very desirable property.

---

# Outputs should be equally strongly typed

Simple:

```typescript
neural function isFraud(
    transaction: Transaction
): boolean
```

Enum:

```typescript
neural function classifyIntent(
    message: string
): Intent
```

where:

```typescript
enum Intent {
    Refund,
    Cancel,
    Support,
    Purchase,
    Other
}
```

But I also want structured outputs.

For example:

```typescript
interface ClaimAssessment {
    category: ClaimCategory;
    severity: Severity;
    fraudRisk: FraudRisk;
    requiresHumanReview: boolean;
}
```

Then:

```typescript
neural function assessClaim(
    claim: Claim,
    customer: Customer
): ClaimAssessment {

    """
    Evaluate the claim.

    Categorize it, assess its severity and fraud risk,
    and determine whether a human needs to review it.
    """
}
```

And ordinary code consumes it:

```typescript
const assessment = assessClaim(claim, customer);

if (assessment.requiresHumanReview) {
    await reviewQueue.add(claim);
}

switch (assessment.severity) {
    case Severity.Critical:
        await alertOnCall(claim);
        break;

    case Severity.Normal:
        await standardQueue.add(claim);
        break;
}
```

No JSON parsing.

No prompt response.

No tool call.

Conceptually, `assessment` is simply another typed variable.

---

# This creates a really interesting compiler opportunity

Consider:

```typescript
const assessment = assessClaim(claim, customer);

if (assessment.requiresHumanReview) {
    ...
}

if (assessment.fraudRisk === FraudRisk.High) {
    ...
}
```

Our compiler can see that `assessClaim()` produces:

```text
ClaimAssessment
├── category
├── severity
├── fraudRisk
└── requiresHumanReview
```

We don't need an autoregressive model to serialize:

```json
{
    "category": "vehicle",
    "severity": "high",
    "fraudRisk": "low",
    "requiresHumanReview": false
}
```

Instead:

```text
                     Shared Encoder
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
          category      severity     fraudRisk
             │             │             │
       categorical    categorical    categorical
             │             │             │
             └─────────────┼─────────────┘
                           │
                           ▼
                  requiresReview
                       boolean
```

All four outputs could potentially be computed **simultaneously**.

That's much closer to what we want.

---

# Inputs can work the same way

We could encode each input according to its type.

```text
assessClaim(
    claim,
    customer,
    explanation
)
```

becomes:

```text
Claim fields ─────────┐
                      │
Customer fields ──────┼──► Encoder
                      │
Explanation text ─────┘
```

Different types could have different representations:

```text
string        → text encoder
enum          → embedding
boolean       → binary representation
integer       → numeric representation
float         → numeric representation
date          → temporal representation
Money         → numeric + currency
object        → field embeddings
array<T>      → sequence/set encoder
```

This is another place where we may be able to outperform a conventional LLM.

A normal LLM sees:

```text
"order_total": "123.45"
```

as tokens.

Our runtime knows:

```text
123.45 : Money<USD>
```

There's no reason to turn everything into text.

---

# Multiple outputs should also be possible

I wouldn't force everything into an object.

Something like:

```typescript
neural function analyzeCustomerRequest(
    customer: Customer,
    message: string
): (
    Intent,
    Sentiment,
    Urgency
) {

    """
    Analyze the customer's request.
    """
}
```

Then:

```typescript
const [intent, sentiment, urgency] =
    analyzeCustomerRequest(customer, message);
```

Which internally becomes three parallel output heads.

---

# And variables can flow between neural and deterministic functions

This is where the language starts becoming powerful.

```typescript
const order = await db.orders.get(id);

const age = daysBetween(
    order.createdAt,
    now()
);

const customer = await db.customers.get(
    order.customerId
);

const decision = evaluateRefund(
    customer,
    order,
    age,
    ctx.body.reason
);

if (decision === RefundDecision.Approve) {

    const amount = calculateRefund(order);

    await db.transaction(async () => {
        await payments.refund(amount);
        await orders.markRefunded(order.id);
    });
}
```

The data flow becomes:

```text
        Postgres
           │
           ▼
        order
           │
           ├─────────────┐
           ▼             ▼
   deterministic      customer
   daysBetween()         │
           │             │
           ▼             │
          age            │
           │             │
           └──────┬──────┘
                  ▼
           neural function
                  │
                  ▼
        RefundDecision variable
                  │
                  ▼
             normal code
                  │
                  ▼
             transaction
```

There isn't really an "AI boundary" from the programmer's perspective.

It's just another function call.

---

## But I'd add one more concept: confidence

Because neural variables are probabilistic, I'd make confidence accessible without contaminating the normal syntax.

Normal:

```typescript
const decision = evaluateRefund(
    customer,
    order
);
```

When needed:

```typescript
const result = evaluateRefund.withConfidence(
    customer,
    order
);

result.value
result.confidence
```

Maybe:

```typescript
if (result.confidence < 0.999) {
    return manualReview(order);
}

if (result.value === RefundDecision.Approve) {
    return refund(order);
}
```

Or a declarative contract:

```typescript
@confidence(0.999)
neural function evaluateRefund(
    customer: Customer,
    order: Order
): RefundDecision {

    """
    ...
    """
}
```

Then the runtime refuses to produce an ordinary `RefundDecision` below the required confidence threshold and invokes the configured fallback.

---

# Now we actually have the beginnings of a language specification

I'd define a neural function as:

```text
NeuralFunction<I, O>

I = finite set of typed inputs
O = finite typed output structure
B = natural-language behavioral specification
C = deterministic constraints
E = optional examples
Q = confidence/calibration requirements
```

So source code:

```typescript
@confidence(0.999)
neural function evaluateRefund(
    customer: Customer,
    order: Order,
    explanation: string
): RefundDecision {

    """
    Apply the company's refund policy.

    Consider the customer's account history,
    the age and status of the order, and the
    customer's explanation.

    Ambiguous or suspicious cases should be
    sent for review rather than denied.
    """

    examples {
        enterpriseOrder45Days => Approve;
        normalOrder45Days     => Deny;
    }
}
```

compiles into something more like:

```text
NEURAL_FUNCTION #17

INPUTS
────────────────────────
customer     Customer
order        Order
explanation  String

OUTPUT
────────────────────────
RefundDecision {
    APPROVE
    DENY
    REVIEW
}

MODEL
────────────────────────
encoder: app_encoder
adapter: refund_17
head: categorical[3]

RUNTIME
────────────────────────
confidence >= .999

TRAINING
────────────────────────
generated:    182,421
adversarial:   41,822
examples:           2

VERIFICATION
────────────────────────
accuracy:      99.997%
type errors:         0
```

And now controllers become almost secondary.

The fundamental thing we're designing is a **new function type**:

> deterministic functions are implemented with instructions; neural functions are implemented with learned behavior.

Inputs and outputs make those two worlds composable. Once we have that abstraction right, controllers, services, CLI programs, event consumers, batch jobs, and potentially frontend code could all use the exact same primitive.

<a id="turn-11"></a>
## User

Since we are mixing with typescript, this no longer needs to be limited to backend work. This allows opens us up to use libraries.

## Assistant

Exactly. That changes the scope considerably.

The interesting thing is no longer a "neural backend." It's a **TypeScript-compatible hybrid programming model** where learned functions and deterministic functions are equally composable.

And keeping TypeScript/JavaScript as the host ecosystem means we don't have to rebuild the world around it.

```typescript
import { readFile } from "node:fs/promises";
import postgres from "postgres";
import Stripe from "stripe";
import { neural } from "@our/runtime";

const stripe = new Stripe(process.env.STRIPE_KEY!);

neural function classifyInvoice(
    invoice: Invoice,
    history: PaymentHistory
): InvoiceAction {
    """
    Determine how this invoice should be handled according
    to our collections policy and the customer's history.
    """
}

const invoice = await getInvoice(id);

// ordinary npm library
const history = await stripe.customers.listPaymentMethods(...);

// learned computation
const action = classifyInvoice(invoice, history);

// ordinary TypeScript
if (action === InvoiceAction.Charge) {
    await stripe.paymentIntents.create(...);
}
```

That is much more compelling.

## TypeScript becomes our systems layer

We don't need the neural runtime to understand Postgres, HTTP, files, React, queues, cryptography, image processing, AWS, etc.

The JavaScript ecosystem already does.

```text
                    TypeScript
                        │
        ┌───────────────┼────────────────┐
        │               │                │
        ▼               ▼                ▼
      npm libs      normal code      neural fn
        │               │                │
        ▼               ▼                ▼
    Postgres          logic          learned logic
    Stripe            math
    AWS               loops
    React             algorithms
    Redis
    Kafka
```

Our contribution becomes surprisingly small conceptually:

> **Add a new kind of function to TypeScript.**

That's a much cleaner project.

---

## And inputs/outputs make libraries naturally interoperable

Suppose an npm package gives us:

```typescript
const image: Buffer = await sharp("photo.jpg")
    .resize(1024, 1024)
    .toBuffer();
```

Then:

```typescript
neural function inspectDamage(
    image: Image,
    claim: Claim
): DamageAssessment {
    """
    Assess the visible damage.

    Determine the damage category, approximate severity,
    and whether the photograph appears consistent with
    the submitted claim.
    """
}
```

Then back into conventional libraries:

```typescript
const assessment = inspectDamage(image, claim);

await db.assessments.insert(assessment);

if (assessment.requiresInspection) {
    await queue.send({
        claimId: claim.id,
        reason: assessment.reason
    });
}
```

That creates:

```text
sharp
  ↓
Buffer
  ↓
neural function
  ↓
DamageAssessment
  ↓
Postgres / queue / API
```

The model doesn't need tools.

**The program has tools.**

That's a really important distinction from agents.

---

# This also solves a major LLM problem

Current agent systems tend to do:

```text
LLM
 │
 ├── decide what to do
 ├── decide which tool
 ├── construct arguments
 ├── invoke tool
 ├── parse result
 ├── decide next tool
 └── repeat
```

We're proposing:

```text
PROGRAM
 │
 ├── deterministic code
 │
 ├── library
 │
 ├── neural function
 │
 ├── deterministic code
 │
 ├── library
 │
 └── neural function
```

The programmer controls the execution graph.

AI provides **semantic computation**, not autonomous orchestration.

That should make the resulting system substantially easier to reason about.

---

# Think about frontend code

Now this works perfectly well in React:

```typescript
function SupportInbox() {

    const messages = useMessages();

    const classifications =
        classifyMessages(messages);

    return (
        <Inbox>
            {messages.map((message, i) => (
                <Message
                    message={message}
                    urgency={classifications[i].urgency}
                    sentiment={classifications[i].sentiment}
                />
            ))}
        </Inbox>
    );
}
```

Where:

```typescript
neural function classifyMessages(
    messages: Message[]
): MessageClassification[] {

    """
    For each customer message determine:

    - urgency
    - sentiment
    - primary intent
    - whether a human response is required
    """
}
```

Potentially compiled to WASM/WebGPU and running **locally in the browser**.

Then:

```text
React
   │
   ▼
neural function
   │
   ▼
WebGPU
   │
   ▼
typed variables
   │
   ▼
React
```

No server inference at all.

That's an entirely different class of application.

---

## Node gives us another huge advantage

We can decide where the same neural function executes.

For example:

```typescript
@runtime("local")
neural function classify(...): Category
```

or:

```typescript
@runtime("gpu")
neural function classify(...): Category
```

or:

```typescript
@runtime("browser")
neural function classify(...): Category
```

or potentially:

```typescript
@runtime("remote")
neural function classify(...): Category
```

Same type signature.

Same behavior.

Different execution target.

Very similar conceptually to compilation targets.

---

# Libraries can also feed neural functions

Imagine using ordinary libraries for preprocessing:

```typescript
import pdf from "pdf-parse";

const document = await pdf(buffer);

const pages = splitPages(document);
```

Then:

```typescript
neural function analyzeContract(
    pages: Page[]
): ContractAnalysis {

    """
    Identify unusual liability terms, termination
    conditions and obligations.
    """
}
```

Then:

```typescript
const analysis = analyzeContract(pages);

const report = generatePDF(analysis);

await s3.putObject(...);
```

Everything except the semantic judgment remains conventional software.

---

# And neural functions themselves become libraries

This might be even more important.

Imagine npm packages containing compiled neural functions:

```bash
npm install @neural/security
```

Then:

```typescript
import {
    classifyLog,
    detectSuspiciousRequest,
    classifyIAMChange
} from "@neural/security";
```

Those aren't API wrappers.

They contain:

```text
package
├── index.d.ts
├── runtime.js
├── security.adapter
└── heads/
    ├── log-classification
    ├── suspicious-request
    └── iam-change
```

And TypeScript sees ordinary signatures:

```typescript
function classifyLog(
    log: SecurityLog
): SecurityEvent;

function detectSuspiciousRequest(
    request: HttpRequest
): ThreatAssessment;
```

That's particularly relevant to the local security-engine concept we discussed earlier: instead of one enormous security LLM reasoning over everything, you could potentially have highly specialized compiled neural functions embedded directly into a streaming pipeline. 

```typescript
kafka.consume("cloud-armor", async event => {

    const threat =
        detectSuspiciousRequest(event);

    if (threat.severity >= Severity.High) {
        await incidents.create(event, threat);
    }
});
```

---

# We could even allow libraries inside neural definitions

There's a subtle but powerful distinction I'd enforce.

This:

```typescript
neural function evaluate(
    transaction: Transaction
): Risk {
    """
    Assess the transaction's fraud risk.
    """
}
```

is learned.

But perhaps this:

```typescript
neural function evaluate(
    transaction: Transaction
): Risk {

    const age = differenceInDays(
        new Date(),
        transaction.createdAt
    );

    const velocity =
        calculateVelocity(transaction);

    """
    Assess fraud risk.

    Account age: ${age}
    Transaction velocity: ${velocity}
    """
}
```

lets deterministic preprocessing occur **inside the logical function**.

The compiler separates it:

```text
evaluate()
    │
    ├── deterministic IR
    │      differenceInDays()
    │      calculateVelocity()
    │
    └── neural IR
           fraud assessment
```

That would let programmers write hybrid functions naturally rather than constantly jumping between two syntaxes.

---

## Which leads to a cleaner syntax idea

Maybe `neural function` isn't actually what we want.

Instead, normal functions could contain **neural expressions**:

```typescript
function evaluateRefund(
    customer: Customer,
    order: Order
): RefundDecision {

    const age =
        differenceInDays(new Date(), order.createdAt);

    if (age > 90)
        return RefundDecision.Deny;

    const suspicious = neural<boolean>`
        Determine whether the circumstances surrounding
        this refund appear suspicious.

        Customer: ${customer}
        Order: ${order}
    `;

    if (suspicious)
        return RefundDecision.Review;

    return neural<RefundDecision>`
        Apply our refund policy to this customer
        and order.
    `;
}
```

Now **functions themselves aren't neural or deterministic**.

Expressions are.

That's probably an even better language model.

```text
TypeScript function
        │
        ├── deterministic expression
        ├── deterministic expression
        ├── neural expression
        ├── control flow
        └── neural expression
```

The compiler extracts the neural expressions, trains them, builds the dependency DAG, and leaves everything else to the normal TypeScript toolchain.

That gives us minimal language surface area while retaining essentially the entire TypeScript/npm ecosystem.

At that point I'd stop calling it a neural backend framework altogether. The concept is closer to a **neural extension to TypeScript**: TypeScript remains TypeScript, npm remains npm, existing applications remain valid, but developers gain one new primitive—**a typed expression whose implementation is learned from a natural-language specification rather than explicitly programmed.**

That's a much larger and, I think, cleaner idea.

<a id="turn-12"></a>
## User

Ok, I need an md file with our investigation and decision points

## Assistant

I handed this off to Work mode to create the Markdown design/investigation document. It will capture the technical investigation, architecture evolution, decisions made, open questions, proposed TypeScript syntax, model/runtime design, training strategy, parallel execution, and an MVP roadmap.

<a id="turn-13"></a>
## User

I need a name for this thing

## Assistant

Yes. Given what this has evolved into, I’d avoid names containing **AI**, **LLM**, or **backend**. The core idea is a programming model where **learned semantic computation becomes a typed primitive alongside deterministic computation**.

My strongest candidates:

- **Synapse** — neural computation embedded into normal programs. `import { neural } from "@synapse/core"`. Strong, but likely crowded.
- **Semant** — from *semantic computation*. `semant<T>`. Feels like a programming-language/toolchain name.
- **Nerve** — neural functions are effectively nerves running through deterministic software. `nerve<T>`. Short and memorable.
- **Cortex** — learned computation layer. Great conceptually, though also crowded.
- **Intuit** — code specifies exact computation; neural expressions provide learned/intuitive computation.
- **Noema** — Greek philosophical term around thought/meaning. Distinctive language/project name.
- **Axiom** — appealing because deterministic constraints establish the axioms within which learned computation operates.
- **Weave** — deterministic and learned computation woven into the same program.
- **Flux** — emphasizes values flowing between deterministic and neural expressions.
- **Sema** — from semantics/semantic. Very compact: `sema<T>`.
- **Neura** — obvious meaning without saying AI.
- **Tessera** — individual neural and deterministic pieces composing into a program.

But I think there are two particularly good naming directions.

### **Semant**

This captures what the new primitive actually provides: **semantic computation**.

The programming model could read:

```typescript
import { semantic } from "@semant/core";

const suspicious = semantic<boolean>`
    Determine whether this transaction
    appears suspicious.
`;
```

And terminology naturally follows:

```text
Semant
Semant compiler
Semant runtime
Semant function
Semant package
Semant model
```

You could describe it in one sentence:

> **Semant adds learned semantic computation as a typed primitive to TypeScript.**

That's pretty clean.

### **Nerve**

This one has stronger branding.

```typescript
import { neural } from "@nerve/core";

const intent = neural<Intent>`
    Determine what this customer wants.
`;
```

And the architecture terminology is excellent:

```text
Nerve Compiler
Nerve Runtime
Nerve IR
Nerve Model
Nerve Package
```

Conceptually:

> **Nerve lets programs think where traditional code can't express the answer precisely.**

The file extension could even eventually be:

```text
application.nerve
```

There's also a third direction I like if we want it to feel more like a **new programming concept** than an AI company:

### **Sema**

Very short, based on the Greek *sēma* ("sign") and closely associated with semantics.

```typescript
const intent = sema<Intent>`
    Determine the customer's intent.
`;
```

That's unusually natural as syntax:

```typescript
const fraud = sema<FraudRisk>`...`;

const category = sema<Category>`...`;

const decision = sema<Decision>`...`;
```

So my current shortlist would be:

**Semant** — strongest technical/product name  
**Nerve** — strongest brand  
**Sema** — strongest language primitive

I especially like the possibility that the **project is Semant** and the actual TypeScript primitive is **`sema<T>`**:

```typescript
import { sema } from "@semant/core";

function processClaim(claim: Claim) {

    const risk = sema<Risk>`
        Assess the fraud risk of this claim.
    `;

    if (risk === Risk.High)
        return manualReview(claim);

    return approve(claim);
}
```

**Semant: typed semantic computation for TypeScript.**

That describes what we're building without tying the project to today's terminology like LLMs or generative AI.

<a id="turn-14"></a>
## User

What about SemantScript?

## Assistant

**SemantScript is strong.** I actually prefer it to just Semant for what we're describing.

It immediately communicates two things: **semantic computation + programming language**, and it fits the TypeScript relationship nicely:

> JavaScript → TypeScript → **SemantScript**

The name also leaves room for this to eventually become more than a TypeScript library/compiler plugin.

The syntax feels natural:

```typescript
import { sema } from "@semantscript/core";

const risk = sema<Risk>`
    Assess the fraud risk of this transaction.
`;
```

Or if we eventually introduce actual language syntax:

```typescript
semantic function assessRisk(
    transaction: Transaction,
    customer: Customer
): Risk {
    """
    Assess the fraud risk based on the transaction,
    customer history, and account behavior.
    """
}
```

And the ecosystem naming works:

```text
SemantScript
SemantScript Compiler
SemantScript Runtime
SemantScript IR
SemantScript SDK

semantscript build
semantscript train
semantscript test
semantscript run

*.sem.ts
```

I particularly like the positioning:

> **SemantScript is a TypeScript-compatible programming model that makes learned semantic computation a typed language primitive.**

Or shorter:

> **SemantScript — where semantics become executable.**

The latter captures the idea unusually well. You're not prompting an LLM. You're **declaring semantic behavior as part of the program**, and the compiler turns that behavior into executable learned computation.

One naming question we'd eventually want to decide is whether **SemantScript is a TypeScript extension** or **its own language that compiles through TypeScript**. For the MVP, I'd strongly favor the former: ordinary `.ts` plus a `sema<T>` primitive, so we inherit TypeScript, Node, npm, IDEs, debugging, libraries, and tooling instead of rebuilding an ecosystem.

<a id="turn-15"></a>
## User

I need a full unedited transcript of this conversation in an md file

## Assistant

I handed this off to Work mode to create `semantscript-conversation-transcript.md` with the full user/assistant conversation preserved verbatim, including all code blocks and without summaries or edits.

<a id="turn-16"></a>
## User

Continuing from [Fine Tune Backend Models](chatgpt-conversation://6aadee05-d280-83e8-b7db-a5ba8e6a6517): Create a Markdown file containing the full, unedited transcript of the current conversation between the user and assistant, preserving every user and assistant message verbatim and in chronological order. Do not summarize, paraphrase, clean up grammar, omit code blocks, or alter wording. Include clear speaker headings such as `## User` and `## Assistant` before each message. Include the conversation from the user's first message in this thread through the user's latest request for the transcript. Do not include hidden system/developer/tool messages or internal reasoning. Name the file `semantscript-conversation-transcript.md` and deliver it as a downloadable artifact.

