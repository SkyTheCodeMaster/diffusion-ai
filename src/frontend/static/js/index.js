async function load_db_counts() {
  let data;
  try {
    let request = await fetch("/api/srv/get/");
    data = await request.json();
  } catch (e) {
    console.error(e)
  }
  const db_size = document.getElementById("db_size");
  db_size.innerText = format(db_size.innerText, data["db_size"] ? data["db_size"] : "No DB");
}

async function setup() {
  await load_db_counts();
}

if (document.readyState == "loading") {
  document.addEventListener("DOMContentLoaded", setup);
} else {
  setup();
}