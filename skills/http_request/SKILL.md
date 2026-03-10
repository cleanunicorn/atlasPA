# http_request

Make HTTP requests (GET, POST, PUT, DELETE, PATCH) to any URL and return the response.

## When to use
- When the user asks you to call an API or fetch data from a URL
- When you need to interact with a REST API or webhook
- When you want to check the content of a URL

## Input
- `url` (string, required): The full URL to request
- `method` (string, optional): HTTP method — GET, POST, PUT, DELETE, PATCH (default: GET)
- `headers` (object, optional): HTTP headers as key-value pairs
- `body` (string, optional): Request body (for POST/PUT/PATCH). Use JSON string for JSON APIs.
- `timeout` (integer, optional): Timeout in seconds (default: 15)

## Output
Returns the response status code, headers summary, and body (truncated if very long).
Returns an error message if the request fails.

## Notes
- Automatically follows redirects
- Response body is truncated at 10,000 characters to fit in context
