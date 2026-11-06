import json, sys, urllib.request
port, path, out_path, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
mid = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=60))["data"][0]["id"]
convs = [json.loads(l) for l in open(path) if l.strip()][:n]
out = open(out_path, "w")
for conv in convs:
    msgs = [{"role": "system", "content": conv["system"]}]
    for t, user in enumerate(conv["turns"][:3]):
        msgs.append({"role": "user", "content": user})
        body = {"model": mid, "temperature": 0, "max_tokens": 400, "chat_template_kwargs": {"enable_thinking": False},
                "messages": msgs}
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=900))
        text = r["choices"][0]["message"]["content"]
        msgs.append({"role": "assistant", "content": text})
        out.write(json.dumps({"id": f"{conv['id']}-t{t + 1}", "text": text, "tokens": r["usage"]["completion_tokens"],
                              "prompt_tokens": r["usage"].get("prompt_tokens"),
                              "finish": r["choices"][0]["finish_reason"]}) + "\n")
        out.flush()
