import { useEffect, useState } from "react";

export default function App() {
  const [message, setMessage] = useState("Connecting to the API...");

  useEffect(() => {
    const controller = new AbortController();

    async function connect() {
      try {
        const response = await fetch("/api", {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error("API request failed");
        }

        const data: { message: string } = await response.json();
        setMessage(data.message);
      } catch {
        if (!controller.signal.aborted) {
          setMessage("Cannot connect to the API.");
        }
      }
    }

    void connect();
    return () => controller.abort();
  }, []);

  return (
    <main style={{ maxWidth: 800, margin: "4rem auto", padding: "1rem" }}>
      <h1>Main page</h1>
    </main>
  );
}