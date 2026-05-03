--- @module ion7.vendor.json
--- SPDX-License-Identifier: MIT
---
--- JSON façade backed by `lua-cjson`.
---
--- Decode behaviour:
---   - JSON `null` decodes to `cjson.null` (a userdata sentinel).
---     Compare with `v == json.null` to detect.
---
--- Encode behaviour:
---   - Empty Lua tables encode as `{}` (object).
---   - NaN / +Inf / -Inf raise an error rather than producing
---     invalid JSON. Sanitise floats at the call site if needed.
---
--- @usage
---   local json = require "ion7.vendor.json"
---   local t  = json.decode('{"name":"Joi","version":2}')
---   local s  = json.encode({ role = "user", content = "hello" })
---   if t.maybe == json.null then ... end

local cjson = require "cjson"

if cjson.encode_empty_table_as_object then
    cjson.encode_empty_table_as_object(true)
end

return {
    encode = cjson.encode,
    decode = cjson.decode,
    null   = cjson.null,
}
