const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });

export interface StrictJsonParseOptions {
  readonly maximumBytes?: number;
  readonly maximumDepth?: number;
  readonly maximumNodes?: number;
}

export class StrictJsonError extends SyntaxError {
  readonly offset: number;

  constructor(message: string, offset: number) {
    super(`${message} at JSON offset ${String(offset)}`);
    this.name = "StrictJsonError";
    this.offset = offset;
  }
}

export function parseStrictJson(
  bytes: Uint8Array,
  options: StrictJsonParseOptions = {},
): unknown {
  const maximumBytes = options.maximumBytes ?? 8 * 1024 * 1024;
  const maximumDepth = options.maximumDepth ?? 128;
  const maximumNodes = options.maximumNodes ?? 1_000_000;

  if (
    !Number.isSafeInteger(maximumBytes) ||
    maximumBytes < 1 ||
    bytes.byteLength > maximumBytes
  ) {
    throw new StrictJsonError("JSON byte limit exceeded", 0);
  }

  if (!Number.isSafeInteger(maximumDepth) || maximumDepth < 1) {
    throw new RangeError("maximumDepth must be a positive safe integer");
  }

  if (!Number.isSafeInteger(maximumNodes) || maximumNodes < 1) {
    throw new RangeError("maximumNodes must be a positive safe integer");
  }

  if (bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    throw new StrictJsonError("JSON must not contain a byte order mark", 0);
  }

  let text: string;

  try {
    text = decoder.decode(bytes);
  } catch (error) {
    throw new StrictJsonError(
      `JSON is not valid UTF-8${error instanceof Error ? `: ${error.message}` : ""}`,
      0,
    );
  }

  const parser = new StrictJsonParser(text, maximumDepth, maximumNodes);
  return parser.parse();
}

class StrictJsonParser {
  readonly #text: string;
  readonly #maximumDepth: number;
  readonly #maximumNodes: number;
  readonly #numberPattern =
    /-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?/uy;
  #offset = 0;
  #nodes = 0;

  constructor(text: string, maximumDepth: number, maximumNodes: number) {
    this.#text = text;
    this.#maximumDepth = maximumDepth;
    this.#maximumNodes = maximumNodes;
  }

  parse(): unknown {
    this.#skipWhitespace();
    const value = this.#parseValue(0);
    this.#skipWhitespace();

    if (this.#offset !== this.#text.length) {
      this.#fail("unexpected content after the JSON value");
    }

    return value;
  }

  #parseValue(depth: number): unknown {
    this.#nodes += 1;

    if (this.#nodes > this.#maximumNodes) {
      this.#fail("JSON node limit exceeded");
    }

    if (depth > this.#maximumDepth) {
      this.#fail("JSON nesting limit exceeded");
    }

    const character = this.#text[this.#offset];

    if (character === "{") return this.#parseObject(depth + 1);
    if (character === "[") return this.#parseArray(depth + 1);
    if (character === '"') return this.#parseString();
    if (character === "t") return this.#parseKeyword("true", true);
    if (character === "f") return this.#parseKeyword("false", false);
    if (character === "n") return this.#parseKeyword("null", null);
    if (character === "-" || isDigit(character)) return this.#parseNumber();

    this.#fail("expected a JSON value");
  }

  #parseObject(depth: number): Record<string, unknown> {
    this.#offset += 1;
    this.#skipWhitespace();
    const value: Record<string, unknown> = {};
    const names = new Set<string>();

    if (this.#text[this.#offset] === "}") {
      this.#offset += 1;
      return value;
    }

    for (;;) {
      if (this.#text[this.#offset] !== '"') {
        this.#fail("expected an object property name");
      }

      const nameOffset = this.#offset;
      const name = this.#parseString();

      if (names.has(name)) {
        throw new StrictJsonError(
          `duplicate object property ${JSON.stringify(name)}`,
          nameOffset,
        );
      }

      names.add(name);
      this.#skipWhitespace();

      if (this.#text[this.#offset] !== ":") {
        this.#fail("expected ':' after an object property name");
      }

      this.#offset += 1;
      this.#skipWhitespace();
      Object.defineProperty(value, name, {
        configurable: true,
        enumerable: true,
        value: this.#parseValue(depth),
        writable: true,
      });
      this.#skipWhitespace();
      const separator = this.#text[this.#offset];

      if (separator === "}") {
        this.#offset += 1;
        return value;
      }

      if (separator !== ",") {
        this.#fail("expected ',' or '}' in an object");
      }

      this.#offset += 1;
      this.#skipWhitespace();
    }
  }

  #parseArray(depth: number): unknown[] {
    this.#offset += 1;
    this.#skipWhitespace();
    const value: unknown[] = [];

    if (this.#text[this.#offset] === "]") {
      this.#offset += 1;
      return value;
    }

    for (;;) {
      value.push(this.#parseValue(depth));
      this.#skipWhitespace();
      const separator = this.#text[this.#offset];

      if (separator === "]") {
        this.#offset += 1;
        return value;
      }

      if (separator !== ",") {
        this.#fail("expected ',' or ']' in an array");
      }

      this.#offset += 1;
      this.#skipWhitespace();
    }
  }

  #parseString(): string {
    const start = this.#offset;
    this.#offset += 1;
    let value = "";

    while (this.#offset < this.#text.length) {
      const character = this.#text[this.#offset];

      if (character === '"') {
        this.#offset += 1;
        assertUnicodeScalarString(value, start);
        return value;
      }

      if (character === "\\") {
        value += this.#parseEscape();
        continue;
      }

      if (!character || character.charCodeAt(0) < 0x20) {
        this.#fail("unescaped control character in a string");
      }

      value += character;
      this.#offset += 1;
    }

    throw new StrictJsonError("unterminated JSON string", start);
  }

  #parseEscape(): string {
    this.#offset += 1;
    const escape = this.#text[this.#offset];
    this.#offset += 1;

    switch (escape) {
      case '"':
      case "\\":
      case "/":
        return escape;
      case "b":
        return "\b";
      case "f":
        return "\f";
      case "n":
        return "\n";
      case "r":
        return "\r";
      case "t":
        return "\t";
      case "u": {
        const hexadecimal = this.#text.slice(this.#offset, this.#offset + 4);

        if (!/^[A-Fa-f0-9]{4}$/u.test(hexadecimal)) {
          this.#fail("invalid Unicode escape");
        }

        this.#offset += 4;
        return String.fromCharCode(Number.parseInt(hexadecimal, 16));
      }
      default:
        this.#fail("invalid string escape");
    }
  }

  #parseKeyword<T>(keyword: string, value: T): T {
    if (!this.#text.startsWith(keyword, this.#offset)) {
      this.#fail(`expected ${keyword}`);
    }

    this.#offset += keyword.length;
    return value;
  }

  #parseNumber(): number {
    this.#numberPattern.lastIndex = this.#offset;
    const match = this.#numberPattern.exec(this.#text);

    if (!match) {
      this.#fail("invalid JSON number");
    }

    const lexical = match[0];
    this.#offset = this.#numberPattern.lastIndex;
    const value = Number(lexical);

    if (!Number.isFinite(value)) {
      this.#fail("JSON number is not finite");
    }

    return value;
  }

  #skipWhitespace(): void {
    while (
      this.#text[this.#offset] === " " ||
      this.#text[this.#offset] === "\n" ||
      this.#text[this.#offset] === "\r" ||
      this.#text[this.#offset] === "\t"
    ) {
      this.#offset += 1;
    }
  }

  #fail(message: string): never {
    throw new StrictJsonError(message, this.#offset);
  }
}

function isDigit(value: string | undefined): boolean {
  return value !== undefined && value >= "0" && value <= "9";
}

function assertUnicodeScalarString(value: string, offset: number): void {
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);

    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);

      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new StrictJsonError(
          "JSON strings must contain only Unicode scalar values",
          offset,
        );
      }

      index += 1;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new StrictJsonError(
        "JSON strings must contain only Unicode scalar values",
        offset,
      );
    }
  }
}
