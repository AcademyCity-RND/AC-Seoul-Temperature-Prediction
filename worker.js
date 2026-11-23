export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const targetBase = "https://avin1731--ac-seoul-temperature-prediction-flask-app.us-east.modal.run";
    const targetUrl = targetBase + url.pathname + url.search;

    // Forward request to Modal
    const response = await fetch(new Request(targetUrl, {
      method: request.method,
      headers: request.headers,
      body: request.body,
      redirect: "follow",
    }));

    // Return response from Modal
    return response;
  }
};