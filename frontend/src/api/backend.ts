export async function analyzeAudio(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

  const res = await fetch(`${apiBaseUrl}/analyze`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error("Analysis failed");
  }
  return res.json();
}
