const fs = require('fs')

const DESCRIPTION_FILE = 'data/descriptions'
const FILTERED_FILE = 'data/filtered.json'

fs.readFile('data/raw.json', 'utf8', (err, data) => {
  if (err) throw err;
  let packages = data
    .split(/,"([^"]+)":(?={"name":"\1")/)
    .filter(package => package.search(/^{"name":"[^"]+"/) != -1)
    .map(package => {
      try {
        return JSON.parse(package)
      } catch (e) {
        return null
      }
    })
  // descriptions should not be empty
    .filter(package => {
      if (package === null
        || package.name == null
        || package.description == null) return false
      if (package.description.length == 0) return false
      return true
    })
  // descriptions should only have ascii charactors
    .filter(({description}) => /^[\x00-\x7F]*$/.test(description))
  const descriptions = packages.map(package => package.description)
  packages = packages
  // names should only have [a-z_-]
    .filter(({name}) => /^[a-z_-]+$/.test(name))
    .map(package => {
      const {name, description} = package
      const splittedName = name.split(/[-_]/).filter(str => str.length >= 1)
      return [splittedName, description]
    })
  // names should have one or more '-' or '_'
    .filter(([name, description]) => {
      return name.length >= 2
    })
  write(DESCRIPTION_FILE, descriptions.join('\n'))
  write(FILTERED_FILE, JSON.stringify(packages))
  console.log(`success (${packages.length} packages and ${descriptions.length} descriptions)`)
});

function write(filename, data) {
  fs.writeFile(filename, data, err => {
    if (err) {
      console.log(err)
    }
  })
}
