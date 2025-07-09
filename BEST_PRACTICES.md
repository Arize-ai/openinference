# OpenInference Development Best Practices & Methodologies

## Established Development Methodology

### Code Review Standards (Strict Enforcement)
**Priority Order for Code Quality**:
1. **Semantic Conventions Compliance** - Highest priority, exact attribute naming and structure per OpenInference spec
2. **Framework Instrumentation Parity** - Functional equivalence with existing Python implementations  
3. **Readability over Defensive Complexity** - Clean, understandable code preferred over complex edge case handling
4. **Comprehensive Testing Coverage** - VCR-based testing with real API responses for robust edge case validation

### Test-Driven Development Methodology
**Established TDD Patterns**:
- **Red-Green-Refactor Cycles**: Strict adherence with test-first development
- **VCR Testing over Manual Mocking**: Real API responses provide better coverage than synthetic mocks
- **Incremental Implementation**: Basic functionality → complexity → advanced features progression
- **"Green Wall" Safety**: Never proceed without passing tests to maintain development confidence

**VCR Testing Implementation Patterns**:
- **Real API Recording**: Live service calls recorded for comprehensive response format testing
- **Credential Sanitization**: Automatic replacement of sensitive data with deterministic mock values
- **Test Isolation**: Global instrumentation setup prevents module patching conflicts
- **Recording Management**: Dedicated tooling for cleanup, validation, and workflow management

### Incremental Refactoring Methodology
**Proven 7-Step Pattern for Safe Refactoring**:
1. **Extract Helper Function** → 2. **Apply to First Location** → 3. **Test** → 4. **Apply to Additional Locations** → 5. **Test** → 6. **Clean Up** → 7. **Final Test**

**Key Principles**:
- **Small Bounded Changes**: Each step should be easily reversible if problems occur  
- **Test Between Every Step**: Maintain "green wall" throughout refactoring process
- **Helper Function Patterns**: Follow established conventions from other instrumentations
- **Extract → Apply → Test**: Methodical approach prevents regressions and maintains confidence

## JavaScript Implementation Advantages (vs Python)

### Architecture Benefits Identified
1. **Stream Processing**: `stream.tee()` enables parallel processing vs sequential Python accumulators
2. **OpenTelemetry Integration**: Native `_wrap()` method simpler than Python's wrapt decorators
3. **AWS SDK Pattern**: Command pattern provides cleaner interception points than method wrapping  
4. **Module Patching**: Global instrumentation setup cleaner than per-test patching
5. **Async/Promise Model**: Native Promise handling vs Python's context manager patterns

### JavaScript-Specific Testing Patterns
**VCR Library Selection**:
- **Nock over Polly**: Simpler HTTP mocking with better Jest integration
- **Manual Recording Control**: Environment variable-based recording mode switching
- **Stream Handling**: Better support for Node.js Readable streams vs Python iterator protocols

## Dependency Management Philosophy

### Surgical Dependency Addition Strategy
**Lessons from Lockfile Management**:
- **Avoid Cascading Changes**: Single dependency should not trigger 1000+ line lockfile changes
- **Purpose-Driven Selection**: Every dependency must solve specific, measurable problem
- **Version Stability**: Prefer established, stable packages over newest features
- **Minimal Surface Area**: Choose focused libraries over kitchen-sink solutions

**Dependency Evaluation Criteria**:
1. **Necessity**: Is this solving a real problem that can't be solved in-house?
2. **Stability**: Is the package mature with stable API and good maintenance?
3. **Size Impact**: What's the bundle size and dependency tree impact?
4. **Alternatives**: Are there simpler or more established alternatives?

### Proven Package Management Practices
- **pnpm Version Consistency**: Ensure team uses same pnpm version to prevent lockfile churn
- **Surgical Updates**: Add dependencies one at a time with lockfile verification
- **Dependency Justification**: Document why each dependency was chosen over alternatives

## JavaScript Instrumentation Patterns

### AWS SDK v3 Instrumentation Best Practices
**Command Pattern Wrapping**:
```typescript
// Target client.send() method for command-based SDKs
this._wrap(moduleExports.ClientClass.prototype, "send", (original) => {
  return function patched(command) {
    if (command instanceof TargetCommand) {
      return instrumentation._handleCommand(command, original, this);
    }
    return original.apply(this, [command]);
  };
});
```

**Attribute Extraction Organization**:
```typescript
// Request-side extraction (command → attributes)
_extractBaseRequestAttributes()     // Model, system, provider, parameters
_extractInputMessagesAttributes()   // User messages and structured input
_extractInputToolAttributes()       // Tool schema conversion

// Response-side extraction (response + span → void)
_extractOutputMessagesAttributes()  // Assistant messages and output value  
_extractToolCallAttributes()        // Tool calls from response content
_extractUsageAttributes()           // Token counts and usage statistics
```

### VCR Testing Implementation Pattern
**Recording Management Workflow**:
```bash
# Development workflow
npm run test:record           # Validate credentials and record live API calls
npm run test                  # Run tests with recorded responses
npm run test:clear-recordings # Clean up old recordings selectively
npm run test:validate-recordings # Verify recording integrity and security
```

**Test Structure Pattern**:
```typescript
// Global instrumentation setup prevents conflicts
beforeAll(() => {
  instrumentation.enable();
});

// Test-specific recording setup
beforeEach(() => {
  setupTestRecording("test-specific-name");
});
```

## Quality Assurance Standards

### Code Organization Requirements
- **Helper Function Extraction**: Break monolithic methods into focused, testable functions
- **Semantic Convention Adherence**: Exact attribute naming per OpenInference specifications
- **Error Handling**: Graceful degradation with comprehensive logging but no defensive complexity
- **Type Safety**: Full TypeScript coverage with strict compiler settings

### Testing Requirements
- **100% Test Coverage**: Every code path must have corresponding test
- **Snapshot Testing**: Inline snapshots verify exact attribute extraction
- **Real API Integration**: VCR recordings from actual service responses
- **Edge Case Coverage**: Error conditions, missing data, malformed responses

This methodology has been proven effective across multiple instrumentation implementations and should be the standard approach for all future OpenInference JavaScript development.
